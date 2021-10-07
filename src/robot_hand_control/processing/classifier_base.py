import collections
import os
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple


class Classifier(ABC):
    """Basic representation for a post processing classifier."""

    model: Any
    model_path: os.PathLike
    classes: list
    buffer_size: int
    classification_buffer: collections.deque


    @abstractmethod
    def load_model(self, model_path: os.PathLike, classes: list) -> None:
        """Load model from file."""

    @abstractmethod
    def format_data(self, data: Iterable) -> Iterable:
        """Format landmarks from Mediapipe to be classified."""

    @abstractmethod
    def run_classification(self, data: Iterable) -> Tuple[str]:
        """Run inference on formatted data."""

    def get_most_common(self) -> Tuple[str]:
        """Get most common item from classification buffer."""
