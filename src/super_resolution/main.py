"""Command line interface to train / test a model or make predictions."""
import os
from pathlib import Path
from time import time

import click
import torch.backends.cudnn as cudnn

from super_resolution.models.model_enum import (
    get_model_from_enum,
    ModelEnum,
)
from super_resolution.super_resolution_data import SuperResolutionData
from super_resolution.utils import load_config, set_seed

# Set seed for reproducibility.
set_seed(0)

cudnn.benchmark = True  # Better performances.


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
def main(config_path: Path) -> None:  # noqa: max-complexity: 13
    """
    Run super resolution.

    Parameters
    ----------
    config_path: Path
        Configuration path.
    """
    # Loading config.
    config = load_config(config_path)

    # Loading model.
    model_path = next(p.parent for p in Path(__file__).parents if p.name == "src")

    model_enum = ModelEnum[config["model_type"]]
    if config["weights_path"] is not None:
        model = get_model_from_enum(
            model_type=model_enum,
            weights_path=model_path / Path(config["weights_path"]),
        )
    else:
        model = get_model_from_enum(model_type=model_enum, weights_path=None)

    # Running task.
    if config["task"] == "train":
        train_dataset = SuperResolutionData(
            image_folder=os.path.join(*config["paths"]["train_set"].split("/")),
            crop_size=config["train"]["crop_size"],
            crop_type="random",
            jpeg_compression=config["jpeg_compression"],
        )

        val_dataset = SuperResolutionData(
            image_folder=os.path.join(*config["paths"]["val_set"].split("/")),
            crop_type="center",
            jpeg_compression=False,
        )

        # Sanity check: remove incorrect files.
        train_dataset.check_sanity(delete=False)
        val_dataset.check_sanity(delete=False)

        if config["paths"]["from_checkpoint"] is not None:
            from_checkpoint = os.path.join(
                *config["paths"]["from_checkpoint"].split("/")
            )
        else:
            from_checkpoint = None

        # One config file by model type.
        model.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_save_folder=os.path.join(
                *config["paths"]["model_save"].split("/"),
                *config["experiment_name"].split("/"),
            ),
            images_save_folder=os.path.join(
                *config["paths"]["val_images_save"].split("/"),
                *config["experiment_name"].split("/"),
            ),
            experiment_name=os.path.join(*config["experiment_name"].split("/")),
            from_checkpoint=from_checkpoint,
            **config["hyperparameters"],
        )

    if config["task"] == "test":
        if isinstance(config["paths"]["test_set"], str):
            config["paths"]["test_set"] = [config["paths"]["test_set"]]
            config["paths"]["test_images_save"] = [config["paths"]["test_images_save"]]

        for image_folder, images_save_folder in zip(
            config["paths"]["test_set"], config["paths"]["test_images_save"]
        ):
            test_dataset = SuperResolutionData(
                image_folder=image_folder,
                crop_type="center",
                jpeg_compression=False,
            )

            metrics = model.evaluate(
                val_dataset=test_dataset,
                images_save_folder=os.path.join(
                    *images_save_folder.split("/"),
                    *config["experiment_name"].split("/"),
                ),
                epoch=1,  # Naming convention
            )
            try:
                psnr, ssim = metrics
            except ValueError:  # SRRESNET also returns loss
                _, psnr, ssim = metrics
            print(f"{os.path.basename(image_folder)} - PSNR: {psnr}, SSIM: {ssim}")

    if config["task"] == "predict":
        if isinstance(config["paths"]["test_set"], str):
            config["paths"]["test_set"] = [config["paths"]["test_set"]]
            config["paths"]["test_images_save"] = [config["paths"]["test_images_save"]]

        for image_folder, images_save_folder in zip(
            config["paths"]["test_set"], config["paths"]["test_images_save"]
        ):
            test_dataset = SuperResolutionData(
                image_folder=image_folder,
                crop_type=config["crop_type"],
                jpeg_compression=False,
            )

            model.predict(
                test_dataset=test_dataset,
                tile_batch_size=config["tile_batch_size"],
                tile_size=config["tile_size"],
                tile_overlap=config["tile_overlap"],
                force_cpu=config["force_cpu"],
                images_save_folder=os.path.join(
                    *images_save_folder.split("/"),
                    *config["experiment_name"].split("/"),
                ),
                batch_size=config["batch_size"],
            )


if __name__ == "__main__":
    start = time()
    main()
    print(f"Task completed in {time() - start:.2f} seconds.")
