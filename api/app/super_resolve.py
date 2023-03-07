"""Module for super resolution in the app."""
import os
from pathlib import Path
from platform import system
from tempfile import TemporaryDirectory

from PIL import Image

from super_resolution.models.model_enum import (
    get_model_from_enum,
    ModelEnum,
)
from super_resolution.super_resolution_data import SuperResolutionData

Image.MAX_IMAGE_PIXELS = (
    None  # Avoid Pillow DecompressionBomb error when opening too big images
)


def get_prediction(image: Image) -> Image:
    """
    Perform super-resolution on a given image.

    Parameters
    ----------
    image: Image
        Image to be processed.

    Returns
    -------
    Image
        Super resolved image.

    Raises
    ------
    FileNotFoundError
        If the model folder is not found.

    """
    if system().lower() == "windows":  # Local
        model_path = next(p.parent for p in Path(__file__).parents if p.name == "api")
    else:  # Docker image
        model_path = Path("/")

    if not model_path.exists():
        raise FileNotFoundError("Model folder not found.")

    model_enum = ModelEnum["SRGAN"]
    weight_path = model_path / Path("models/SRGAN/generator_epoch_71.torch")

    # Only trained generator not discriminator.
    model = get_model_from_enum(model_type=model_enum, weights_path=weight_path)

    images_save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "predictions"
    )
    with TemporaryDirectory() as dataset_folder:
        image.save(os.path.join(dataset_folder, image.filename), quality=100)
        dataset = SuperResolutionData(
            image_folder=dataset_folder,
            scaling_factor=4,
            crop_type="no_crop",
        )
        try:
            print("Using GPU", flush=True)
            model.predict(
                test_dataset=dataset,
                images_save_folder=images_save_folder,
                force_cpu=False,
                tile_batch_size=2,
                tile_size=128,
                tile_overlap=10,
                batch_size=1,
            )
        except (
            RuntimeError,
            OSError,
        ):  # CUDA out of memory: try to predict using CPU only.
            print("Not enough GPU memory: will run on CPU.", flush=True)
            model.predict(
                test_dataset=dataset,
                images_save_folder=images_save_folder,
                force_cpu=True,
                tile_batch_size=8,
                tile_size=128,
                tile_overlap=10,
                batch_size=1,
            )

    return Image.open(os.path.join(images_save_folder, "0.png"))
