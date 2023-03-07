"""Test cases for the __main__ module."""
from super_resolution.models.model_enum import get_model_from_enum, ModelEnum
from super_resolution.models.SRGAN.model import SRGAN
from super_resolution.models.SRResNet.model import SRResNet


def test_get_model_from_enum() -> None:
    """Testing get_model_from_enum function."""
    srgan = get_model_from_enum(model_type=ModelEnum["SRGAN"])
    srresnet = get_model_from_enum(model_type=ModelEnum["SRRESNET"])

    assert isinstance(srgan, SRGAN)
    assert isinstance(srresnet, SRResNet)
