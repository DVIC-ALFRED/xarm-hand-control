import collections
import os
from statistics import mode
from typing import Iterable, List, Tuple

import numpy as np
import onnxruntime
from xarm_hand_control.processing.classifier_base import Classifier


class ONNXMLP(Classifier):
    """Classifier with a Multi-layer Percptron using ONNX."""

    model: onnxruntime.InferenceSession
    model_path: os.PathLike
    classes: list
    buffer_size: int
    classification_buffer: collections.deque
    input_name: str
    output_name: str

    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.classification_buffer = collections.deque(maxlen=buffer_size)

    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        self.model_path = model_path
        self.classes = classes

        self.model = onnxruntime.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

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

    def run_classification(self, data: List[np.ndarray]) -> Tuple[str]:
        ret = []

        for item in data:
            onnx_outputs = self.model.run([self.output_name], {self.input_name: item})
            result_one = np.argmax(onnx_outputs[0].squeeze(axis=0))

            ret.append(self.classes[result_one]["name"])

        ret = tuple(ret)

        self.classification_buffer.appendleft(ret)

        return ret

    def get_most_common(self) -> Tuple[str]:
        return mode(self.classification_buffer)
