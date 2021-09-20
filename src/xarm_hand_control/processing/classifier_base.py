import os
from abc import ABC, abstractmethod
from typing import Iterable, Tuple


class Classifier(ABC):
    """Basic representation for a post processing classifier."""

    @abstractmethod
    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        """Load model from file."""

    @abstractmethod
    def format_data(self, data: Iterable) -> Iterable:
        """Format landmarks from Mediapipe to be classified."""

    @abstractmethod
    def run_classification(self, data: Iterable) -> Tuple[dict]:
        """Run inference on formatted data."""
