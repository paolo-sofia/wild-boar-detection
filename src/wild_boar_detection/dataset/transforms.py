import random

import torch


class StereoToMono(torch.nn.Module):
    """Convert audio from stereo to mono."""

    def __init__(self, reduction: str = "avg", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(reduction, str)
        assert reduction in {"avg", "sum"}
        self.reduction = reduction

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.squeeze()
        if sample.shape[0] == 1 or len(sample.shape) == 1:
            return sample
        return sample.mean(dim=0) if self.reduction == "avg" else sample.sum(dim=0)


class AudioCrop(torch.nn.Module):
    def __init__(self, sample_rate: int, crop_size: int = 30) -> None:
        super().__init__()
        self.crop_size: int = crop_size  # in seconds
        self.sample_rate: int = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.crop_size * self.sample_rate:
            return x

        start_frame: torch.Tensor = torch.randint(
            low=0, high=max(0, x.shape[1] - (self.crop_size * self.sample_rate)), size=(1,)
        ).detach()
        return x[:, start_frame : start_frame + (self.crop_size * self.sample_rate)]


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma: float = 25.0, p: float = 1.0) -> None:
        super().__init__()
        assert 0 <= p <= 1
        self.sigma: float = sigma
        self.p: float = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x

        return torch.clip(x + self.sigma * torch.randn_like(x))


class Squeeze(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)


class MinMaxNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.new_min: float = 0.0
        self.new_max: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        epsilon: torch.tensor = torch.tensor(1e-15, dtype=x.dtype)
        # print(f"Min Max input shape: {x.shape}")

        return ((x - x_min) / (max(x_max - x_min, epsilon))) * (self.new_max - self.new_min) + self.new_min
