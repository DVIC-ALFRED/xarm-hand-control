import os
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from xarm_hand_control.processing.classifier_base import Classifier


class RandomForest(Classifier):
    """Classifier with a Multi-layer Percptron using ONNX."""

    model_path: os.PathLike
    classes: list

    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        self.model_path = model_path
        self.classes = classes

        self.model = joblib.load(self.model_path)

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
                    ]
                ).reshape(1, -1)
            )

        return ret

    def run_classification(self, data: List[np.ndarray]) -> Tuple[dict]:
        ret = []

        for item in data:
            result_one = self.model.predict(item)[0]

            ret.append({"index": result_one, "class_name": self.classes[result_one]["name"]})

        return tuple(ret)
