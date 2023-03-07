"""Utility functions for super resolution."""
import os
from pathlib import Path
import random
from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from super_resolution.super_resolution_data import SuperResolutionData

ROOT_DIR = next(p.parent for p in Path(__file__).parents if p.name == "src")


def set_seed(seed: int = 0) -> None:
    """
    Set seed for reproducibility.

    Parameters
    ----------
    seed: int
        The seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_stat(
    dataset: Union[SuperResolutionData, str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an RGB image dataset mean and standard deviation for normalization purpose.

    Parameters
    ----------
    dataset: Union[SuperResolutionData, str]
        The dataset to be analyzed. Can be a SuperResolutionData dataset or a string (only 'ImageNet' supported).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        List of mean for each channel, list of standard deviation for each channel.

    Raises
    ------
    ValueError
        If 'dataset' is not 'ImageNet'.

    """
    if isinstance(dataset, str):
        if dataset.lower() == "imagenet":
            return torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor(
                [0.229, 0.224, 0.225]
            )

        raise ValueError(
            f"Parameter 'dataset' as a string should be 'ImageNet' but is {dataset}"
        )

    data_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4
    )
    total_batch = len(data_loader)

    channels_sum, channels_squared_sum, num_batches = (
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    for i_batch, (_, hr_images) in enumerate(data_loader):
        print(f"{i_batch + 1}/{total_batch}\r", end="")

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(hr_images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(hr_images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def load_config(config_path: Path) -> Dict:
    """
    Load a configuration file.

    Parameters
    ----------
    config_path: Path
        The config file path.

    Returns
    -------
    Dict
        The configuration object.
    """
    with open(config_path, encoding="utf8") as file:
        config = yaml.safe_load(file)
    return config
