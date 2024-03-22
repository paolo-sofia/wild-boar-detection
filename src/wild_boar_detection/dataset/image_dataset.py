import logging
import os
import pathlib
from typing import Self

import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms import v2

from wild_boar_detection.dataset.transforms import AddGaussianNoise
from wild_boar_detection.utils import get_dir_absolute_path


class ImageDataset(torch.utils.data.Dataset):
    """Dataset class for audio data.

    Args:
        data_path: The path to the audio data.
        mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
    """

    def __init__(
        self: Self,
        data_path: pathlib.Path | str,
        mode: str = "train",
    ) -> None:
        """Initializes the AudioDataset with the specified parameters.

        Args:
            data_path: The path to the audio data.
            mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
        """
        __slots__ = ["data_path", "mode", "precision", "transforms"]
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: pd.DataFrame = pd.read_parquet(data_path)
        self.precision: torch.dtype = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }.get(int(os.getenv("PRECISION", "16")))

        self.transforms = self._init_transforms(mode)
        self.base_data_path: pathlib.Path = get_dir_absolute_path("data").parent

    def _init_transforms(self: Self, mode: str) -> v2.Compose:
        image_size: int = int(os.getenv("IMAGE_SIZE", "256"))

        if mode == "train":
            return v2.Compose(
                [
                    AddGaussianNoise(),
                    v2.RandomResizedCrop(size=(image_size, image_size), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(degrees=(-180, 180)),
                    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                    v2.RandomAutocontrast(p=0.1),
                    v2.RandomEqualize(p=0.1),
                    v2.RandomGrayscale(p=0.5),
                    v2.RandomApply(
                        transforms=[
                            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                        ]
                    ),
                    # v2.ToImage(),
                    v2.ToDtype(self.precision, scale=True),
                    # v2.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
                ]
            )
        return v2.Compose(
            [
                v2.Resize(size=(image_size, image_size), antialias=True),
                # v2.ToImage(),
                v2.ToDtype(self.precision, scale=True),
                # v2.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            ]
        )

    def __len__(self: Self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_path)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the item at the specified index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        try:
            img = read_image(str(self.base_data_path / self.data_path.loc[index, "path"]))
        except Exception as e:
            logging.error(self.base_data_path / self.data_path.loc[index, "path"])
            logging.error(e)
            raise Exception(e)

        return self.transforms(img), self.data_path.loc[index, "target"], self.data_path.loc[index, "weight"]
