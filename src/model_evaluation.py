import pathlib
from time import sleep
from typing import Generator

import cv2
import numpy as np
import onnxruntime

MODEL_PATH: pathlib.Path = pathlib.Path(
    "../data/logs/mlruns/669180362677009476/28c74b00b3c24059a887a64895e6dedf/artifacts/model/model.onnx"
).resolve()
TEST_SET_PATHS: Generator[pathlib.Path, None, None] = pathlib.Path("../data/test_set/").glob("*")

with pathlib.Path(
    "../data/logs/mlruns/669180362677009476/28c74b00b3c24059a887a64895e6dedf/metrics/threshold"
).resolve().open("r") as f:
    THRESHOLD = round(float(f.readline().split(" ")[1]), 2)


ort_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(MODEL_PATH)
input_name: str = ort_session.get_inputs()[0].name

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

for video_path in TEST_SET_PATHS:
    vidcap = cv2.VideoCapture(str(video_path))
    frame_counter = 1
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (1000 * frame_counter))  # added this line

        success: bool
        image: np.ndarray
        success, image = vidcap.read()

        if not success:
            break

        inputs: np.ndarray = cv2.resize(image, (256, 256))
        inputs = inputs.transpose(2, 0, 1)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs.astype(np.float32)
        inputs /= 255.0

        outputs: list[np.array] = ort_session.run(None, {input_name: inputs})

        y: float = 1 / (1 + np.exp(-outputs[0].item()))
        y_class: bool = y >= THRESHOLD

        cv2.putText(
            img=image,
            text=f"Is wild boar? {y_class} - Prediction: {y:.3f}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )

        # Display the frame
        cv2.imshow("Video", cv2.resize(image, (640, 480)))

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        sleep(2)

    vidcap.release()

    cv2.destroyAllWindows()
