"""Module defining class to store data or super resolution model."""
from io import BytesIO
import os
from random import randrange
from time import time
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Lambda,
    PILToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)


class SuperResolutionData(Dataset):
    """
    Class for data loader.

    Each High Resolution image is cropped to a squared size, then downscaled to make the Low resolution image.
    """

    def __init__(
        self,
        image_folder: str,
        crop_type: str,
        crop_size: Optional[int] = None,
        scaling_factor: int = 4,
    ) -> None:
        """
        Initialize data class.

        Parameters
        ----------
        image_folder: str
            A folder containing images.
        crop_type: str
            The type of crop. Should be in
              - "center": for test and val split, takes the largest possible (squared) center-crop.
              - "random": for train split, takes random crop of size crop_size.
              - "no_crop": for prediction, allows to predict non-squared images.
        crop_size: int
            The target size for high resolution images for train set. Test set: images are not cropped to a fixed size.
        scaling_factor: int
            The scaling factor to use when downscaling high resolution images into low resolution images.

        Raises
        ------
        ValueError
            If the crop_size is not divisible by scaling_factor, or crop_type not in 'random', 'center', 'no_crop',
            or crop_type is 'random' but no crop_size is specified.

        """
        if crop_size is not None and crop_size % scaling_factor != 0:
            raise ValueError(
                "Crop size is not divisible by scaling factor! This will lead to a mismatch in the \
                              dimensions of the original high resolution patches and their super-resolved \
                              (reconstructed) versions!"
            )

        crop_type = crop_type.lower()
        if crop_type == "random" and crop_size is None:
            raise ValueError("As crop_type is 'random', 'crop_size' must be specified")

        if crop_type not in ("random", "center", "no_crop"):
            raise ValueError(
                "crop_type value must be in 'random', 'center', or 'no_crop'"
            )

        self.images_path = [
            os.path.join(image_folder, image_name)
            for image_name in os.listdir(image_folder)
        ]
        self.image_folder = image_folder
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.scaling_factor = scaling_factor

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get preprocessed images, high and low resolution for the model.

        Note: range of output LR images is [0, 1], HR images in [-1, 1] (cf https://arxiv.org/pdf/1609.04802.pdf).

        Parameters
        ----------
        idx: int
            index of the element to get

        Returns
        -------
        Tuple[tensor, tensor]
            The low resolution image and the high resolution image
        """
        image = Image.open(self.images_path[idx])
        image = image.convert("RGB")

        # 1. Crop original image to make the High Resolution image (in [0, 1]).
        if self.crop_type == "random":  # For training
            transform_hr = Compose(
                [
                    RandomCrop(self.crop_size),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                ]
            )
        elif (
            self.crop_type == "center"
        ):  # For evaluation and testing. No random transformations (reproducible).
            # Take the largest possible (squared) center-crop such that dimensions are divisible by the scaling factor.
            original_width, original_height = image.size
            crop_size = min(
                (
                    original_height - original_height % self.scaling_factor,
                    original_width - original_width % self.scaling_factor,
                )
            )
            transform_hr = Compose(
                [
                    CenterCrop(crop_size),
                ]
            )
        else:
            transform_hr = Compose([])

        # 2. Downscale image to  make the Low Resolution image.
        high_res_image = transform_hr(image)
        high_res_height, high_res_width = high_res_image.size

        if (
            self.crop_type == "random"
        ):  # Jpeg compression only applied on training images.
            transform_lr = Compose(
                [
                    Resize(
                        (
                            high_res_height // self.scaling_factor,
                            high_res_width // self.scaling_factor,
                        ),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    Lambda(
                        random_jpeg_compression
                    ),  # Random JPEG compression applied on LR PIL image.
                ]
            )

        else:
            transform_lr = Compose(
                [
                    Resize(
                        (
                            high_res_height // self.scaling_factor,
                            high_res_width // self.scaling_factor,
                        ),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                ]
            )
        low_res_image = transform_lr(high_res_image)

        # Convert PIL image to torch tensor: swap axis from (H x W x C) to (C x H x W).
        low_res_image = PILToTensor()(low_res_image)
        high_res_image = PILToTensor()(high_res_image)

        # Scale tensor range.
        low_res_image = low_res_image / 255.0  # Output range in [0, 1]
        high_res_image = (high_res_image / 255.0) * 2.0 - 1.0  # Output range in [-1, 1]

        return low_res_image, high_res_image

    def __len__(self) -> int:
        """
        Return the dataset size.

        Returns
        -------
        int
            Number of images in the dataset.
        """
        return len(self.images_path)

    def check_sanity(self, delete: bool = False) -> None:
        """
        Check dataset sanity before training.

        All image sizes must be above crop size if specified.
        All images must be valid image files to be read by Pillow.

        Parameters
        ----------
        delete: bool
            Whether to delete invalid image files or not
        """
        start = time()
        total_images = len(self)
        for index, image_path in enumerate(self.images_path):
            print(
                f"{index + 1}/{total_images} "
                f'[{"=" * int(40 * (index + 1) / total_images)}>'
                f'{"-" * int(40 - 40 * (index + 1) / total_images)}] '
                f"- Duration {time() - start:.1f} s\r",
                end="",
            )

            try:
                image = Image.open(image_path)  # UnidentifiedImageError
                image = image.convert(
                    "RGB"
                )  # OSError: broken data stream when reading image file
            except (UnidentifiedImageError, OSError):
                print(f"Image {image_path} is not a valid image file.")
                self.images_path.remove(image_path)
                if delete:
                    os.remove(image_path)
            else:
                if self.crop_type == "random" and (
                    image.size[0] < self.crop_size or image.size[1] < self.crop_size
                ):
                    print(f"Image {image_path} is too small for cropping {image.size}")
                    image.close()
                    self.images_path.remove(image_path)
                    if delete:
                        os.remove(image_path)


def random_jpeg_compression(image: Image) -> Image:
    """
    Apply random jpeg compression to an image.

    Parameters
    ----------
    image: Image
        Input image to be compressed.

    Returns
    -------
    Image
        The compressed image.
    """
    quality = randrange(50, 100)
    output_stream = BytesIO()
    image.save(output_stream, "JPEG", quality=quality, optimize=True)
    output_stream.seek(0)
    return Image.open(output_stream)
