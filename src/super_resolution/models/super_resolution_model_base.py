"""Abstract class for model."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from super_resolution.models.SRGAN.generator import Generator
from super_resolution.super_resolution_data import SuperResolutionData


class SuperResolutionModelBase(ABC):
    """Abstract class for models."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a super resolution model."""

    @abstractmethod
    def train(
        self,
        train_dataset: SuperResolutionData,
        val_dataset: SuperResolutionData,
        epochs: int,
        experiment_name: str,
        model_save_folder: str,
        images_save_folder: str,
        batch_size: int,
        learning_rate: float,
        from_checkpoint: Optional[str],
    ) -> None:
        """Define template for training method."""

    @abstractmethod
    def evaluate(
        self,
        val_dataset: SuperResolutionData,
        epoch: int,
        images_save_folder: str,
    ) -> Tuple:
        """Define template for evaluation method."""

    @abstractmethod
    def load(self, generator: Union[Generator, Path]) -> None:
        """Define template for model loading method."""

    @abstractmethod
    def predict(
        self,
        test_dataset: SuperResolutionData,
        images_save_folder: str,
        batch_size: int,
        force_cpu: bool,
        tile_size: Optional[int],
        tile_overlap: Optional[int],
        tile_batch_size: Optional[int],
    ) -> None:
        """Define template for model prediction method."""
