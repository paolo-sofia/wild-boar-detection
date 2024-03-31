import logging
import os
import pathlib
from io import BytesIO

import numpy as np
import onnxruntime
from dotenv import load_dotenv
from PIL import Image
from robyn import Request, Robyn

load_dotenv()
app = Robyn(__file__)


def load_threshold() -> float:
    with pathlib.Path(os.getenv("THRESHOLD_PATH")).resolve().open("r") as f:
        threshold: float = round(float(f.readline()), 2)

    return threshold


THRESHOLD: float = load_threshold()
ort_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(os.getenv("MODEL_PATH"))
input_name: str = ort_session.get_inputs()[0].name


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"message": "Healthy"}


@app.post("/predict")
async def predict(request: Request) -> dict[str, bool | float] | dict[str, str | Exception]:
    try:
        byte_data: bytes = bytes(request.body)
        start_index: int = byte_data.find(b"\r\n\r\n") + len(b"\r\n\r\n")

        # Extract the image data
        image_data: bytes = byte_data[start_index:]
        inputs: Image.Image = Image.open(BytesIO(image_data))
        inputs = inputs.resize((256, 256))

        inputs: np.ndarray = np.asarray(inputs)
        inputs = inputs.transpose(2, 0, 1)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs.astype(np.float32)
        inputs /= 255.0

        outputs: list[np.array] = ort_session.run(None, {input_name: inputs})

        y_proba: float = (1 / (1 + np.exp(-outputs[0].item()))).item()
        y_class: bool = bool(y_proba >= THRESHOLD)
        return {"class": y_class, "probability": y_proba}

    except Exception as e:
        logging.error(f"Error inside the predict function {e}")
        return {"message": "Error when performing prediction", "error": str(e)}


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8000)
