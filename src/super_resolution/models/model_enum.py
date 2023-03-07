"""List of trained models."""
from enum import Enum
from pathlib import Path
from typing import Optional

from super_resolution.models.SRGAN.discriminator import Discriminator
from super_resolution.models.SRGAN.generator import Generator
from super_resolution.models.SRGAN.model import SRGAN
from super_resolution.models.SRResNet.model import SRResNet
from super_resolution.models.super_resolution_model_base import (
    SuperResolutionModelBase,
)


class ModelEnum(Enum):
    """Enum with models that we can use for super resolution."""

    SRGAN = (0,)
    SRRESNET = (1,)
    BASELINE = (2,)


def get_model_from_enum(
    model_type: ModelEnum, weights_path: Optional[Path] = None
) -> SuperResolutionModelBase:
    """
    Get a specific model from enum.

    We never need to load the trained discriminator of a SRGAN model.

    Parameters
    ----------
    weights_path: Optional[Path]
        Model folder path, needed as models are loaded from various locations. Default None.
    model_type: ModelEnum
        The model to be loaded.

    Returns
    -------
    SuperResolutionModelBase
        The loaded model.

    Raises
    ------
    ValueError
        If the model_type is not implemented, which means not defined in enum.
    FileNotFoundError
        If the model_path does not exist.
    """
    if model_type.name.lower() == "srgan":
        srgan = SRGAN(discriminator=Discriminator(), generator=Generator())
        if weights_path is not None:
            try:
                srgan.load(
                    generator=weights_path
                )  # We never need to load pretrained discriminator for SRGAN model.
            except FileNotFoundError:
                raise FileNotFoundError(f"Model path {weights_path} not found.")

        return srgan

    if model_type.name.lower() == "srresnet":
        srresnet = SRResNet()
        if weights_path is not None:
            try:
                srresnet.load(generator=weights_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Model name {weights_path} not defined in enum"
                )
        return srresnet

    raise ValueError(f"Model type {model_type} not defined in enum.")
