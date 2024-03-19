import random
from copy import deepcopy
from typing import Self

import numpy as np
import torch
import torchaudio.transforms as T
from safetensors import safe_open
from torch import nn
from torch_audiomentations import OneOf
from torchvision.transforms import v2

from .transforms import AddGaussianNoise, MinMaxNorm


class AudioDataset(torch.utils.data.Dataset):
    """Dataset class for audio data.

    Args:
        data_path: The path to the audio data.
        crop_size: The size of the cropped audio in seconds.
        mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
    """

    def __init__(
        self: Self,
        data_path: np.ndarray | list[str],
        crop_size: int,
        precision: int,
        mel_spectrogram_param: dict[str, int],
        mode: str = "train",
    ) -> None:
        """Initializes the AudioDataset with the specified parameters.

        Args:
            data_path: The path to the audio data.
            crop_size: The size of the cropped audio in seconds.
            mode: The mode of the dataset, either "train", "valid", or "test". Defaults to "train".
        """
        assert mode in {"train", "valid", "test"}
        super().__init__()
        self.data_path: np.ndarray | list[str] = data_path
        self.crop_size: int = crop_size
        self.mode: str = mode
        self.mel_spectrogram_param: dict[str, int] = mel_spectrogram_param
        self.precision: torch.dtype = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }.get(precision)
        self.sample_rate: int = 48000

        self._init_transforms()

    def _init_transforms(self: Self) -> None:
        transforms: list[nn.Module] = [
            T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.mel_spectrogram_param.get("N_FFT", 512),
                win_length=self.mel_spectrogram_param.get("WIN_LENGTH", 512),
                hop_length=self.mel_spectrogram_param.get("HOP_LENGTH", 256),
                n_mels=self.mel_spectrogram_param.get("N_MELS", 256),
                normalized=False,
            ),
            MinMaxNorm(),
            # v2.Resize(
            #     size=(self.mel_spectrogram_param.get("N_MELS", 256), 5626)
            # ),  # needed because some samples can be of less than 30 seconds or have a different frame rate
            v2.ToDtype(self.precision, scale=False),
        ]

        self.y_transforms = v2.Compose(transforms)
        train_transforms = deepcopy(transforms)

        if self.mode == "train":
            train_transforms.insert(1, AddGaussianNoise(p=1.0))
            train_transforms.insert(
                2, OneOf([T.TimeMasking(time_mask_param=100), T.FrequencyMasking(freq_mask_param=30)])
            )

        self.x_transforms = v2.Compose(train_transforms)

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
        with safe_open(self.data_path[index], framework="pt", device="cpu") as f:
            audio: torch.Tensor = f.get_tensor("audio")

        num_frames: int = audio.shape[1]
        crop_frames: int = self.crop_size * self.sample_rate
        if num_frames <= crop_frames:
            frames_to_add: int = crop_frames - audio.shape[1]
            audio = torch.cat([audio, torch.zeros((audio.shape[0], frames_to_add))], dim=1)
            return self.x_transforms(audio), self.y_transforms(audio)

        x_transformed: torch.Tensor
        y_transformed: torch.Tensor
        while True:
            frame_offset: int = random.randint(0, num_frames - crop_frames)
            cropped_audio = audio[:, frame_offset : frame_offset + crop_frames]
            x_transformed, y_transformed = self.x_transforms(cropped_audio), self.y_transforms(cropped_audio)
            if not torch.isnan(x_transformed).sum() and not torch.isnan(x_transformed).sum():
                break
        return x_transformed, y_transformed
