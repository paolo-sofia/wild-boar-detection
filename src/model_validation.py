import pathlib
from time import sleep

import cv2
import numpy as np
import onnxruntime
import pandas as pd

MODEL_PATH: pathlib.Path = pathlib.Path(
    "../data/logs/mlruns/669180362677009476/28c74b00b3c24059a887a64895e6dedf/artifacts/model/model.onnx"
).resolve()

with pathlib.Path(
    "../data/logs/mlruns/669180362677009476/28c74b00b3c24059a887a64895e6dedf/metrics/threshold"
).resolve().open("r") as f:
    THRESHOLD = round(float(f.readline().split(" ")[1]), 2)

BASE_DATA_PATH = pathlib.Path("/home/paolo/git/wild-boar-detection")

dataframe: pd.DataFrame = pd.read_parquet(pathlib.Path("../data/valid.parquet").resolve())

ort_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(MODEL_PATH)
input_name: str = ort_session.get_inputs()[0].name


cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

predictions: dict[str, list[bool | float]] = {"y_true": [], "y_pred": [], "y_prob": []}

for target in [0, 1]:
    for _, row in dataframe[dataframe.target == target].iterrows():
        image = cv2.imread(str(BASE_DATA_PATH.joinpath(row.path)))

        inputs: np.ndarray = cv2.resize(image, (256, 256))
        inputs = inputs.transpose(2, 0, 1)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs.astype(np.float32)
        inputs /= 255.0

        outputs: list[np.array] = ort_session.run(None, {input_name: inputs})

        y: float = 1 / (1 + np.exp(-outputs[0].item()))
        y_class: bool = y >= THRESHOLD

        predictions["y_true"].append(bool(target))
        predictions["y_pred"].append(y_class)
        predictions["y_prob"].append(y)

        cv2.putText(
            img=image,
            text=f"Is wild boar? {y_class} - Prediction: {y:.3f}. True target: {bool(target)}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )

        # Display the frame
        cv2.imshow("Video", cv2.resize(image, (1920, 10qq80)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        sleep(2)

        cv2.destroyAllWindows()


predictions: pd.DataFrame = pd.DataFrame(data=predictions)
predictions.to_csv("predictions.csv", index=False)
