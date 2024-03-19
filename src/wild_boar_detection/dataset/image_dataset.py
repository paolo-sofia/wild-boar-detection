from typing import Self

import numpy as np
import pandas as pd
import torch


class ImageDataset(torch.utils.data.Dataset):
    """Dataset class for audio data.

    Args:
        data_path: The path to the audio data.
        crop_size: The size of the cropped audio in seconds.
        mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
    """

    def __init__(
        self: Self,
        data_path: np.ndarray | list[str],
        mode: str = "train",
        precision: int = 16,
    ) -> None:
        """Initializes the AudioDataset with the specified parameters.

        Args:
            data_path: The path to the audio data.
            mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
            precision: Float precision used for training.
        """
        __slots__ = ["data_path", "mode", "precision"]
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: pd.DataFrame = pd.read_parquet(data_path)
        self.mode: str = mode
        self.precision: torch.dtype = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }.get(precision)

        self._init_transforms()

    def _init_transforms(self: Self) -> None:
        pass

    def __len__(self: Self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_path)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the item at the specified index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        # print(self.data_path[index])
        # load image
        # transform image
        # return image
