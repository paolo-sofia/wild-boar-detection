use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, read::DecoderReader};
use dotenv::dotenv;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::any::type_name;
use std::f32::consts::E;
use std::{
    env,
    fs::File,
    io::prelude::*,
    io::{self, BufRead, Cursor},
    path::Path,
    process,
};
use tower_http::limit::RequestBodyLimitLayer;
use tract_onnx::{prelude::*, tract_hir::ops::math::mul};

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

#[derive(Deserialize)]
struct InputImage {
    image_base64: String,
}

// the input to our `create_user` handler
#[derive(Serialize)]
struct Prediction {
    class: bool,
    probability: f32,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn decode_base64_image(base64_image: &str) -> Result<Vec<u8>, image::ImageError> {
    let mut cursor = Cursor::new(base64_image);

    let mut decoder = DecoderReader::new(&mut cursor, &STANDARD);

    let mut image = Vec::new();
    decoder.read_to_end(&mut image).unwrap();

    Ok(image)
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

    println!("Length of image is {} bytes", data.len());

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
    println!("original prediction: {result:?}");
    // find and display the max value with its index
    let proba = match result[0].to_scalar::<f32>() {
        Ok(proba) => proba,
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
    };
    let proba_f64: f64 = (*proba).into();
    let probability = 1.0 / (1.0 + E.powf(-proba));
    println!("result: {proba:?}");
    let class = probability > state.threshold;
    println!("class: {class:?}");
    let prediction = Prediction { class, probability };

    (StatusCode::CREATED, Json(prediction)).into_response()
}

fn preprocess_image(image: &DynamicImage) -> TractResult<Tensor> {
    // Resize image to 256x256
    let resized_image = image.resize_exact(256, 256, image::imageops::FilterType::Nearest);

    // Convert to RGB if not already in RGB
    let rgb_image = match resized_image {
        DynamicImage::ImageRgb8(_) => resized_image,
        _ => resized_image.into_rgb8().into(),
    };

    // Convert the image into a tensor
    let tensor_data: Vec<f32> = rgb_image
        .as_bytes()
        .iter()
        .map(|&x| x as f32 / 255.0) // Normalize the pixel values
        .collect();

    // Convert the image into a tensor
    let mut tensor = Tensor::from_shape(&[3, 256, 256], &tensor_data)?;

    let result = tensor.insert_axis(0);
    result?;
    // let mut reshaped_tensor = tensor.into_shape((3, 256, 256)).unwrap();
    // tensor = tensor.insert_axis(0)?;
    Ok(tensor)

    // let normalized_tensor = reshaped_tensor.map(|x| x / 255.0);

    // normalized_tensor
}

async fn health_check() -> &'static str {
    "Hello, World!"
}

async fn make_prediction<
    M: std::borrow::Borrow<
        tract_onnx::prelude::Graph<
            tract_onnx::prelude::TypedFact,
            std::boxed::Box<(dyn tract_onnx::prelude::TypedOp + 'static)>,
        >,
    >,
>(
    State(state): State<Arc<AppState<M>>>,
    Json(payload): Json<InputImage>,
) -> Response<Body> {
    //(StatusCode, Json<Prediction>)
    let decoded_image = match decode_base64_image(&payload.image_base64) {
        Ok(decoded_image) => decoded_image,
        Err(_) => {
            eprintln!("Error while decoded_image");
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

    let img = match image::load_from_memory(&decoded_image) {
        Ok(img) => img,
        Err(_) => {
            eprintln!("Error while decoded_image");
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
            eprintln!("Error while decoded_image");
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

    let prediction = Prediction {
        class: true,
        probability: 1.0,
    };

    // run the model on the input
    let result = &state.model.run(tvec!(tensor.into()));

    // find and display the max value with its index
    let best = match result {
        Ok(result) => result,
        Err(_) => {
            eprintln!("Error while decoded_image");
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

    println!("result: {best:?}");
    (StatusCode::CREATED, Json(prediction)).into_response()
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

    let mut threshold: f32 = 0.5;
    let mut index: i8 = 0;
    if let Ok(lines) = read_lines(threshold_path) {
        for line in lines.flatten() {
            if index != 1 {
                index = index + 1;
                continue;
            }

            let splitted_line: Vec<&str> = line.split_whitespace().collect();
            if let Some(second_value) = splitted_line.get(1) {
                if let Ok(threshold_value) = second_value.parse::<f32>() {
                    threshold = threshold_value;
                    break;
                } else {
                    println!("Failed to parse second value as float: {}", second_value);
                    break;
                }
            } else {
                println!("Line does not have a second value");
                break;
            }
        }
    } else {
        println!("Error while loading threshold from file. Setting threshold value to 0.5");
    }

    let model_path = match env::var("MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Error: MODEL_PATH environment variable not found");
            process::exit(1);
        }
    };

    println!("model path {}", model_path);

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
