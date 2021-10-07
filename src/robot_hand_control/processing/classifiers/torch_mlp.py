import collections
import os
from statistics import mode
from typing import Iterable, List, Tuple

import torch
from xarm_hand_control.processing.classifier_base import Classifier
from xarm_hand_control.training.model import HandsClassifier


class TorchMLP(Classifier):
    """Classifier with a Multi-layer Percptron using PyTorch."""

    model: torch.nn.Module
    model_path: os.PathLike
    classes: list
    buffer_size: int
    classification_buffer: collections.deque
    n_classes: int

    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.classification_buffer = collections.deque(maxlen=buffer_size)

    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        self.model_path = model_path
        self.classes = classes
        self.n_classes = len(self.classes)

        self.model = HandsClassifier(self.n_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def format_data(self, data: Iterable) -> List[torch.Tensor]:
        ret = []

        for hand_landmarks in data:
            hand_landmarks_xys = [
                [point.x, point.y] for point in hand_landmarks.landmark
            ]

            ret.append(
                torch.tensor(
                    [
                        hand_landmarks_xys,
                    ]
                )
            )

        return ret

    def run_classification(self, data: List[torch.Tensor]) -> Tuple[str]:
        ret = []

        for item in data:
            result_one = torch.argmax(self.model(item)).item()

            ret.append(self.classes[result_one]["name"])

        ret = tuple(ret)

        self.classification_buffer.appendleft(ret)

        return ret

    def get_most_common(self) -> Tuple[str]:
        return mode(self.classification_buffer)
