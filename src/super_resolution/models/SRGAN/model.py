"""SRGAN model implementation."""
from math import ceil
import os
from pathlib import Path
from time import time
from typing import Optional, Tuple, Union

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Resize,
)
from torchvision.utils import make_grid, save_image

from super_resolution.models.SRGAN.discriminator import Discriminator
from super_resolution.models.SRGAN.generator import Generator
from super_resolution.models.SRGAN.utils_loss import TruncatedVGG
from super_resolution.models.super_resolution_model_base import (
    SuperResolutionModelBase,
)
from super_resolution.models.utils_models import (
    get_image_from_tiles,
    get_tiles_from_image,
    RGB_WEIGHTS,
)
from super_resolution.super_resolution_data import SuperResolutionData


class SRGAN(SuperResolutionModelBase):
    """Super resolution with GAN model."""

    def __init__(self, discriminator: Discriminator, generator: Generator) -> None:
        """
        Initialize the truncated VGG network.

        It is used to compute the perceptual loss is only loaded when training SRGAN model to avoid unnecessary model
        loading.

        Parameters
        ----------
        discriminator: Discriminator
            The discriminator model. Classify images as real or fake.
        generator: Generator
            The generator model. Generates fake images.
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator

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
        beta_loss: float = 0.001,
    ) -> None:
        """
        Train and save the final model.

        Divide learning rate by 10 at mid training.

        Parameters
        ----------
        train_dataset: SuperResolutionData
            Dataset to use for training.
        val_dataset: SuperResolutionData
            Dataset used for validation.
        epochs: int
            Number of epochs.
        experiment_name: str
            Name of the experiment.
        model_save_folder: str
            Folder where to save checkpoints.
        images_save_folder: str
            Folder where to save validation images (low, high and super resolution) at each validation step.
        batch_size: int
            Batch size for training.
        learning_rate: float
            Learning rate used for training.
        from_checkpoint: Optional[str]
            If provided, path to resume the training from previous checkpoint.
        beta_loss: float
            The coefficient to weight the adversarial loss in the perceptual loss.
        """
        # Create log file to monitor training and evaluation.
        writer = SummaryWriter(os.path.join("logs", experiment_name))

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.generator.to(device)
        self.discriminator.to(device)

        # Load the truncated VGG network used to compute the perceptual loss.
        truncated_vgg = TruncatedVGG()
        truncated_vgg.to(device)
        truncated_vgg.eval()  # Used to compute the content loss only.

        data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

        # Initialize generator's optimizer
        optimizer_g = Adam(
            params=filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=learning_rate,
        )
        scheduler_g = MultiStepLR(optimizer_g, milestones=[epochs // 2], gamma=0.1)

        # Initialize discriminator's optimizer
        optimizer_d = Adam(
            params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=learning_rate,
        )
        scheduler_d = MultiStepLR(optimizer_d, milestones=[epochs // 2], gamma=0.1)

        start_epoch = 0
        if from_checkpoint is not None:
            checkpoint = torch.load(from_checkpoint)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
            scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
            start_epoch = (
                checkpoint["epoch"] + 1
            )  # Start at the next epoch to avoid overwriting.

        if os.path.exists(model_save_folder):
            # If loaded from checkpoint, epoch number is saved: checkpoints will have different names and won't be
            # overridden.
            print(f"Warning ! {model_save_folder} already exists.")
        else:
            os.makedirs(model_save_folder)

        # Generator loss
        content_loss = MSELoss().to(device)

        # Discriminator loss
        adversarial_loss = BCEWithLogitsLoss().to(device)

        start = time()

        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch}/{epochs}")

            running_adversarial_loss_d = 0.0
            running_adversarial_loss_g = 0.0
            running_vgg_loss_g = 0.0
            running_pixel_loss_g = 0.0
            running_perceptual_loss_g = 0.0

            self.generator.train()
            self.discriminator.train()

            total_batch = len(data_loader)

            # lr/hr/sr: low/high/super resolution
            # Training step.
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                lr_images = lr_images.to(device)  # in [0, 1]
                hr_images = hr_images.to(device)  # in [-1, 1]

                sr_images = self.generator(
                    lr_images
                )  # Super resolution images in [-1, 1].

                # GENERATOR
                # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
                # VGG network contain imageNET normalization layer.
                sr_images_vgg = truncated_vgg(sr_images)
                hr_images_vgg = truncated_vgg(
                    hr_images
                ).detach()  # Detach as they don't need gradient (targets).

                # Images discrimination (on SR images from generator output).
                sr_discriminated = self.discriminator(sr_images)  # (N)

                # Calculate the Perceptual loss using MSE on VGG space and MSE on pixel space.
                vgg_loss_g = 0.001 * 0.1 * content_loss(sr_images_vgg, hr_images_vgg)
                pixel_loss_g = 0.9 * content_loss(sr_images, hr_images)
                adversarial_loss_g = beta_loss * adversarial_loss(
                    sr_discriminated, torch.ones_like(sr_discriminated)
                )
                # Cf https://iopscience.iop.org/article/10.1088/1742-6596/1903/1/012050/pdf .
                perceptual_loss_g = vgg_loss_g + pixel_loss_g + adversarial_loss_g

                # Backward step: compute gradients.
                perceptual_loss_g.backward()

                # Step: update model parameters.
                optimizer_g.step()
                self.generator.zero_grad(set_to_none=True)

                # DISCRIMINATOR
                hr_discriminated = self.discriminator(hr_images)
                sr_discriminated = self.discriminator(sr_images.detach().clone())
                # Don't use previous sr_discriminated because it would also update generator parameters.

                d_sr_probability = torch.sigmoid_(
                    torch.mean(sr_discriminated.detach())
                )  # prob of sr
                d_hr_probability = torch.sigmoid_(
                    torch.mean(hr_discriminated.detach())
                )  # prob of hr

                # # Binary Cross-Entropy loss
                adversarial_loss_d = adversarial_loss(
                    sr_discriminated, torch.zeros_like(sr_discriminated)
                ) + adversarial_loss(
                    hr_discriminated, torch.ones_like(hr_discriminated)
                )

                # Backward step: compute gradients.
                adversarial_loss_d.backward()

                # Step: update model parameters.
                optimizer_d.step()
                self.discriminator.zero_grad(set_to_none=True)

                print(
                    f"\r{i_batch}/{total_batch} "
                    f'[{"=" * int(40 * i_batch  / total_batch)}>'
                    f'{"-" * int(40 - 40 * i_batch / total_batch)}] '
                    f"- Loss generator {perceptual_loss_g.item():.4f} "
                    f"- Loss discriminator {adversarial_loss_d.item():.4f} "
                    f"- Duration {time() - start:.1f} s",
                    end="",
                )

                # Save logs for tensorboard.
                iteration = i_batch + epoch * total_batch
                writer.add_scalar(
                    "Train/discriminator_total_loss",
                    adversarial_loss_d.item(),
                    iteration,
                )
                writer.add_scalar(
                    "Train/generator_vgg_loss", vgg_loss_g.item(), iteration
                )

                writer.add_scalar(
                    "Train/generator_pixel_loss", pixel_loss_g.item(), iteration
                )

                writer.add_scalar(
                    "Train/generator_adversarial_loss",
                    adversarial_loss_g.item(),
                    iteration,
                )
                writer.add_scalar(
                    "Train/generator_total_loss",
                    perceptual_loss_g.item(),
                    iteration,
                )
                writer.add_scalar(
                    "Train/discriminator_hr_probability", d_hr_probability, iteration
                )
                writer.add_scalar(
                    "Train/discriminator_sr_probability", d_sr_probability, iteration
                )

                running_adversarial_loss_d += adversarial_loss_d.item()
                running_adversarial_loss_g += adversarial_loss_g.item()
                running_vgg_loss_g += vgg_loss_g.item()
                running_pixel_loss_g += pixel_loss_g.item()
                running_perceptual_loss_g += perceptual_loss_g.item()

            scheduler_g.step()
            scheduler_d.step()

            # Evaluation step.
            psnr, ssim = self.evaluate(
                val_dataset,
                images_save_folder=images_save_folder,
                epoch=epoch,
            )

            writer.add_scalar("Val/PSNR", psnr, epoch)
            writer.add_scalar("Val/SSIM", ssim, epoch)

            print(
                f"Epoch {epoch}/{epochs} "
                f"- Loss discriminator (adversarial): {running_adversarial_loss_d / total_batch:.4f} "
                f"- Loss generator (adversarial): {running_adversarial_loss_g / total_batch:.4f} "
                f"- Loss generator (vgg): {running_vgg_loss_g / total_batch:.4f} "
                f"- Loss generator (pixel): {running_pixel_loss_g / total_batch:.4f} "
                f"- Loss generator (total): {running_perceptual_loss_g / total_batch:.4f} "
                f"- PSNR: {psnr:.2f} "
                f"- SSIM: {ssim:.2f} "
                f"- Duration {time() - start:.1f} s"
            )

            # Save training state to pause or resume the training.
            torch.save(
                {
                    "generator_state_dict": self.generator.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict(),
                    "scheduler_d_state_dict": scheduler_d.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(model_save_folder, f"checkpoint_{epoch}.torch"),
            )

            # Free some memory since their histories may be stored
            del (
                lr_images,
                hr_images,
                sr_images,
                hr_images_vgg,
                sr_images_vgg,
                hr_discriminated,
                sr_discriminated,
            )

        return

    def evaluate(
        self,
        val_dataset: SuperResolutionData,
        epoch: int,
        images_save_folder: str,
    ) -> Tuple[float, float]:
        """
        Test the model, using PSNR and SSIM.

        No validation data should be provided as GAN cannot be monitored using a validation loss.
        PSNR [dB] and SSIM measures are calculated on the y-channel of center-cropped, removal of a 4-pixel wide strip
        from each border PSNR is computed on the Y channel (luminance) of the YCbCr image.

        Parameters
        ----------
        val_dataset: SuperResolutionData
            dataset to use for testing.
        epoch: int
            Epoch number used for filename in saved images.
        images_save_folder: str
            Folder to save generated images.

        Returns
        -------
        Tuple[float, float]
            Average PSNR and SSIM values.
        """
        if os.path.exists(images_save_folder):
            print(f"Warning ! {images_save_folder} already exists.")
        else:
            os.makedirs(images_save_folder)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.generator.to(device)

        all_psnr = []
        all_ssim = []

        # No need for ToTensor as already a Tensor Batch * Channel * Height * Width
        transform = Compose([Resize(400), CenterCrop(400)])

        data_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Batch size is 1 as each image have a specific size.
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        total_batch = len(data_loader)

        start = time()

        self.generator.eval()

        with torch.no_grad():
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                lr_images = lr_images.to(device)  # In [0, 1]
                hr_images = hr_images.to(device)  # In [-1, 1]
                sr_images = self.generator(
                    lr_images
                )  # Super resolution images in [-1, 1].

                # Save images.
                if images_save_folder and (
                    total_batch <= 10 or i_batch % (total_batch // 10) == 0
                ):
                    for i in range(sr_images.size(0)):
                        # Each image is scaled into [0, 1]
                        images = torch.stack(
                            [
                                transform(lr_images[i, :, :, :]),
                                transform((sr_images[i, :, :, :] + 1.0) / 2.0),
                                transform((hr_images[i, :, :, :] + 1.0) / 2.0),
                            ]
                        )
                        grid = make_grid(images, nrow=3, padding=5)
                        # Saving a grid for image comparison.
                        save_image(
                            grid,
                            os.path.join(
                                images_save_folder,
                                f"epoch_{epoch}_image_{i_batch + i}.png",
                            ),
                            padding=5,
                        )

                        # Saving also full super resolved image
                        save_image(
                            images[1],
                            os.path.join(
                                images_save_folder,
                                f"epoch_{epoch}_SR_{i_batch + i}.png",
                            ),
                        )

                hr_images = (
                    255.0 * (1.0 + hr_images) / 2.0
                )  # Map from [-1, 1] to [0, 255]
                sr_images = (
                    255.0 * (1.0 + sr_images) / 2.0
                )  # Map from [-1, 1] to [0, 255]

                # Use Y channel only (luminance) to compute PSNR and SSIM (RGB to YCbCr conversion)
                sr_Y = (
                    torch.matmul(
                        sr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                        RGB_WEIGHTS.to(device),
                    )
                    / 255.0
                    + 16.0
                )
                hr_Y = (
                    torch.matmul(
                        hr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                        RGB_WEIGHTS.to(device),
                    )
                    / 255.0
                    + 16.0
                )

                # Change device
                sr_Y = sr_Y.cpu().numpy()
                hr_Y = hr_Y.cpu().numpy()

                # Calculate PSNR
                batch_psnr = [
                    peak_signal_noise_ratio(sr_Y[i], hr_Y[i], data_range=255.0)
                    for i in range(sr_Y.shape[0])
                ]

                # Calculate SSIM
                batch_ssim = [
                    structural_similarity(sr_Y[i], hr_Y[i], data_range=255.0)
                    for i in range(sr_Y.shape[0])
                ]

                all_psnr.extend(batch_psnr)
                all_ssim.extend(batch_ssim)

                print(
                    f"\r{i_batch}/{total_batch} "
                    f'[{"=" * int(40 * i_batch  / total_batch)}>'
                    f'{"-" * int(40 - 40 * i_batch / total_batch)}] '
                    f"- Duration {time() - start:.1f} s",
                    end="",
                )

        average_psnr = np.round(sum(all_psnr) / len(all_psnr), 3)
        average_ssim = np.round(sum(all_ssim) / len(all_ssim), 3)

        return average_psnr, average_ssim

    def predict(
        self,
        test_dataset: SuperResolutionData,
        images_save_folder: str,
        batch_size: int = 1,
        force_cpu: bool = True,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        tile_batch_size: Optional[int] = None,
        scaling_factor: int = 4,
    ) -> None:
        """
        Process an image into super resolution.

        We use high resolution images as input.

        Parameters
        ----------
        test_dataset: SuperResolutionData
            The images to process.
        batch_size: int
            The batch size for predictions. If prediction made by tiles, batch_size should be 1.
        tile_batch_size: Optional[int], default None
            Images are processed one by one, however tiles for a given image can be processed by batches.
        images_save_folder: str
            The folder where to save predicted images.
        force_cpu: bool
            Whether to force usage of CPU or not (inference on high resolution images may run GPU out of memory).
        tile_size: Optional[int], default None
            As too large images result in the out of GPU memory issue, tile option will first crop input images into
            tiles, then process each of them. Finally, they will be merged into one image. None: not using tiles.
            It is advised to use the same tile_size as low resolution image in the training.
            Adapted from https://github.com/ata4/esrgan-launcher/blob/master/upscale.py
        tile_overlap: Optional[int], default None
            Overlap pixels between tiles.
        scaling_factor: int
            The scaling factor to use when downscaling high resolution images into low resolution images.

        Raises
        ------
        ValueError
            If 'tile_size' is not None and 'batch_size' is not 1, as prediction by tiles only supports batch_size of 1.
        """
        if os.path.exists(images_save_folder):
            print(f"Warning ! {images_save_folder} already exists.")
        else:
            os.makedirs(images_save_folder)

        if tile_size is not None and batch_size != 1:
            raise ValueError(
                f"Prediction is made by tile as 'tile_size' is specified. Only batch_size of 1"
                f" is supported, but is '{batch_size}'. To predict tiles by batch, please use "
                f"'tile_batch_size'."
            )

        device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not force_cpu
            else torch.device("cpu")
        )
        self.generator.to(device)
        self.generator.eval()

        # pin_memory can lead to too much pagination memory needed.
        data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        total_batch = len(data_loader)

        start = time()

        with torch.no_grad():
            for i_batch, (_, images) in enumerate(data_loader):
                # As we give hr image as input to the model, we need to scale it from [-1, 1] to [0, 1]
                # like the normal lr_image input.
                images = (images + 1.0) / 2.0

                if (
                    tile_size is not None
                    and tile_overlap is not None
                    and tile_batch_size is not None
                ):
                    image = images[0]  # batch_size must be 1

                    # 1. Get tiles from input image
                    tiles = get_tiles_from_image(image, tile_size, tile_overlap)

                    # Retrieve tiles by row and by col to merge predictions
                    channel, height, width = image.shape
                    tiles_x = ceil(width / tile_size)
                    tiles_y = ceil(height / tile_size)

                    sr_tiles = torch.empty(
                        (
                            tiles_x * tiles_y,
                            channel,
                            scaling_factor * (tile_size + 2 * tile_overlap),
                            scaling_factor * (tile_size + 2 * tile_overlap),
                        )
                    )

                    # 2. Loop over all tiles to make predictions
                    batches = torch.split(
                        tiles, tile_batch_size, dim=0
                    )  # Create batches of tiles
                    total_batch_tile = len(batches)

                    for i_batch_tile, batch in enumerate(batches):
                        print(
                            f"\r{i_batch_tile}/{total_batch_tile} "
                            f'[{"=" * int(40 * i_batch_tile / total_batch_tile)}>'
                            f'{"-" * int(40 - 40 * i_batch_tile / total_batch_tile)}] '
                            f"- Duration {time() - start:.1f} s",
                            end="",
                        )
                        index_start = i_batch_tile * tile_batch_size
                        index_end = min(
                            (i_batch_tile + 1) * tile_batch_size, tiles.size()[0]
                        )  # Last batch may be smaller.
                        sr_tiles[index_start:index_end] = self.generator(
                            batch.to(device)
                        )

                    # 3. Merge upscaled tiles: retrieve index and position from indexes
                    sr_images = get_image_from_tiles(
                        sr_tiles, tile_size, tile_overlap, scaling_factor, image
                    )

                else:
                    sr_images = self.generator(images.to(device))

                # Scale image from [-1, 1] to [0,1]
                sr_images = (sr_images + 1.0) / 2.0

                # Save images
                if images_save_folder:
                    for i in range(sr_images.size(0)):
                        save_image(
                            sr_images[i, :, :, :],
                            os.path.join(images_save_folder, f"{i_batch + i}.png"),
                            format="PNG",
                        )

                print(
                    f"\r{i_batch}/{total_batch} "
                    f'[{"=" * int(40 * i_batch / total_batch)}>'
                    f'{"-" * int(40 - 40 * i_batch / total_batch)}] '
                    f"- Duration {time() - start:.1f} s",
                    end="",
                )

        return

    def load(
        self,
        generator: Optional[Union[Generator, Path]] = None,
        discriminator: Optional[Union[Discriminator, Path]] = None,
    ) -> None:
        """
        Load a pretrained model.

        Parameters
        ----------
        generator: Optional[Union[Generator, Path]], default None
            A path to a pretrained model, or a torch model.
        discriminator: Optional[Union[Discriminator, Path]]: default None
            A path to a pretrained model, or a torch model.

        Raises
        ------
        TypeError
            If 'generator' or 'discriminator' type is not a string or a model.

        """
        if generator is not None:
            if isinstance(generator, Path):
                self.generator.load_state_dict(torch.load(generator))  # From path
            elif isinstance(generator, type(self.generator)):
                self.generator.load_state_dict(generator.state_dict())
            else:
                raise TypeError(
                    "Generator argument must be either a path to a trained model, or a trained model."
                )

        if discriminator is not None:
            if isinstance(discriminator, Path):
                self.discriminator.load_state_dict(
                    torch.load(discriminator)
                )  # From path
            elif isinstance(discriminator, Discriminator):
                self.discriminator.load_state_dict(discriminator.state_dict())
            else:
                raise TypeError(
                    "Generator argument must be either a path to a trained model, or a trained model."
                )
