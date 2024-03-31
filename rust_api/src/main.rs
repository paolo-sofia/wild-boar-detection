use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dotenv::dotenv;
use image::DynamicImage;
use serde::Serialize;
use std::{
    env,
    fs,
    f32::consts::E,
    io::Error,
    process,
};
use tower_http::limit::RequestBodyLimitLayer;
use tract_onnx::prelude::*;

struct AppState<
    M: std::borrow::Borrow<
        tract_onnx::prelude::Graph<
            tract_onnx::prelude::TypedFact,
            std::boxed::Box<(dyn tract_onnx::prelude::TypedOp + 'static)>,
        >,
    >,
> {
    model: TypedRunnableModel<M>,
    threshold: f32,
}

// the input to our `create_user` handler
#[derive(Serialize)]
struct Prediction {
    class: bool,
    probability: f32,
}

fn read_float_from_file(file_path: &str) -> Result<f32, Error> {
    // Open the file
    let contents = fs::read_to_string(file_path).expect("Error while reading threshold file: {file_path}");
    let th = match contents.trim().parse::<f32>() {
        Ok(th) => th,
        Err(_) => {
                      eprintln!("Error: MODEL_PATH environment variable not found");
                      0.5
                  }
    };
    Ok(th)
}


async fn predict_image<
    M: std::borrow::Borrow<
        tract_onnx::prelude::Graph<
            tract_onnx::prelude::TypedFact,
            std::boxed::Box<(dyn tract_onnx::prelude::TypedOp + 'static)>,
        >,
    >,
>(
    State(state): State<Arc<AppState<M>>>,
    mut multipart: Multipart,
) -> Response<Body> {
    let mut data = Vec::new();

    while let Some(file) = multipart.next_field().await.unwrap() {
        let chunk = file.bytes().await.unwrap();
        data.extend_from_slice(&chunk);
    }

    let img = match image::load_from_memory(&data) {
        Ok(img) => img,
        Err(_) => {
            eprintln!("Error loading image from memory");
            let response = Prediction {
                class: false,
                probability: -1.0,
            };
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(response),
            )
                .into_response();
        }
    };

    let tensor = match preprocess_image(&img) {
        Ok(tensor) => tensor,
        Err(_) => {
            eprintln!("Error while preprocess_image");
            let response = Prediction {
                class: false,
                probability: -1.0,
            };
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(response),
            )
                .into_response();
        }
    };

    // run the model on the input
    let result = &state.model.run(tvec!(tensor.into()));
    let result = match result {
        Ok(result) => result,
        Err(_) => {
            eprintln!("Error predict_image");
            let response = Prediction {
                class: false,
                probability: -1.0,
            };
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(response),
            )
                .into_response();
        }
    }; //.unwrap().remove(0);

    let proba = match result[0].to_scalar::<f32>() {
        Ok(proba) => proba,
        Err(_) => {
            eprintln!("Error parsing model outputs");
            let response = Prediction {
                class: false,
                probability: -1.0,
            };
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(response),
            )
                .into_response();
        }
    };

    let probability: f32 = 1.0 / (1.0 + E.powf(-proba));
    let class: bool = probability > state.threshold;
    let prediction: Prediction = Prediction { class, probability };

    (StatusCode::CREATED, Json(prediction)).into_response()
}

fn preprocess_image(image: &DynamicImage) -> TractResult<Tensor> {
    let resized_image = image.resize_exact(256, 256, image::imageops::FilterType::Nearest);

    let rgb_image = match resized_image {
        DynamicImage::ImageRgb8(_) => resized_image,
        _ => resized_image.into_rgb8().into(),
    };

    let tensor_data: Vec<f32> = rgb_image
        .as_bytes()
        .iter()
        .map(|&x| x as f32 / 255.0) // Normalize the pixel values
        .collect();

    let mut tensor = Tensor::from_shape(&[3, 256, 256], &tensor_data)?;

    let result = tensor.insert_axis(0);
    result?;
    Ok(tensor)
}

async fn health_check() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    dotenv().ok();

    let threshold_path = match env::var("THRESHOLD_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Error: THRESHOLD_PATH environment variable not found");
            process::exit(1);
        }
    };

    let threshold: f32 = match read_float_from_file(&threshold_path) {
            Ok(value) => value,
            Err(_) => {
                println!("Error while loading threshold from file. Setting threshold value to 0.5");
                0.5 // Default threshold value if reading fails
            }
        };

    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Error: MODEL_PATH environment variable not found");
            process::exit(1);
        }
    };

    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .expect("REASON")
        .into_optimized()
        .unwrap_or_else(|err| {
            eprintln!("Error while loading model: {err}");
            process::exit(1);
        });

    let typed_model = model.into_runnable().expect("Porcodio");
    println!("");

    let shared_state = Arc::new(AppState {
        model: typed_model,
        threshold,
    });

    let app = Router::new()
        .route("/", get(health_check))
        .route("/predict", post(predict_image))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(25 * 1024 * 1024 /* 25mb */))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
