import os
from typing import Iterable, List, Tuple

import numpy as np
import onnxruntime
from xarm_hand_control.processing.classifier_base import Classifier


class ONNXMLP(Classifier):
    """Classifier with a Multi-layer Percptron using ONNX."""

    model_path: os.PathLike
    classes: list
    session: onnxruntime.InferenceSession
    input_name: str
    output_name: str

    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        self.model_path = model_path
        self.classes = classes

        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def format_data(self, data: Iterable) -> List[np.ndarray]:
        ret = []

        for hand_landmarks in data:
            hand_landmarks_xys = [
                [point.x, point.y] for point in hand_landmarks.landmark
            ]

            ret.append(
                np.array(
                    [
                        hand_landmarks_xys,
                    ],
                    dtype=np.float32,
                )
            )

        return ret

    def run_classification(self, data: List[np.ndarray]) -> Tuple[dict]:
        ret = []

        for item in data:
            onnx_outputs = self.session.run([self.output_name], {self.input_name: item})
            result_one = np.argmax(onnx_outputs[0].squeeze(axis=0))

            ret.append({"index": result_one, "class_name": self.classes[result_one]["name"]})

        return tuple(ret)
