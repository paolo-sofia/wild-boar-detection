from __future__ import annotations

from typing import Self

import lightning
import torch
from torch import nn, optim

from .optimizers.yogi import Yogi


class Autoencoder(lightning.LightningModule):
    """Initializes the Autoencoder model with the specified hyperparameters.

    Args:
        hyperparam: The hyperparameters of the autoencoder.
        loss: The loss function used for training the autoencoder.
    """

    def __init__(
        self: Self,
        model,
        loss: nn.Module = nn.BinaryCrossEntropyLoss,
    ) -> None:
        """Initializes the Autoencoder model.

        Args:
            hyperparam: The hyperparameters of the autoencoder.
            loss: The loss function used for training the autoencoder.

        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.loss = loss(reduction="mean")

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the network and returns the output.

        Args:
            inputs: The input tensor to be passed through the network.

        Returns:
            The output tensor produced by the network.
        """
        return self.model(inputs)

    def configure_optimizers(self: Self) -> dict[str, optim.Optimizer | optim.lr_scheduler.LRScheduler | str]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            A dictionary containing the optimizer, learning rate scheduler, and monitor metric.
        """
        optimizer: optim.Optimizer = Yogi(
            params=self.parameters(), lr=self.hyperparam.LEARNING_RATE, weight_decay=self.hyperparam.LEARNING_RATE_DECAY
        )

        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.hyperparam.T_0, T_mult=self.hyperparam.T_mult
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}

    def _log_metrics(self: Self, loss: float, step: str = "train") -> None:
        """Logs the specified loss metric during training or validation.

        Args:
            loss: The value of the loss metric to be logged.
            step: The step or phase of the training or validation process. Defaults to "train".
        """
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def training_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Performs a training step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="train")

    def validation_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Performs a validation step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="valid")

    def test_step(self: Self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Performs a validation step on a batch of data.

        Args:
            batch: A tuple containing the input tensors for the validation step.
            batch_idx: The index of the current batch.

        Returns:
            The output of the forward pass on the batch.
        """
        return self._forward(batch, step="test")

    def _forward(self: Self, batch: tuple[torch.Tensor, ...], step: str) -> torch.Tensor:
        """Performs a forward pass on a batch of data and calculates the loss.

        Args:
            batch: The input batch of data.
            step: The step or phase of the forward pass.

        Returns:
            The calculated loss value.
        """
        x, y = batch
        x_hat = self.forward(x)
        loss = self.loss(x_hat, y)
        self._log_metrics(loss=loss, step=step)
        return loss


def initialize_weights(layer: nn.Module) -> None:
    """Initializes the weights of the network.

    Args:
        layer: The layer of the network.
    """
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_uniform_(layer.weight.data, a=0.0003, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight.data, 1)
        nn.init.constant_(layer.bias.data, 0)
