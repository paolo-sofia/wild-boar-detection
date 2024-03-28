from __future__ import annotations

import inspect
import logging
from typing import Callable, Self

import lightning
import timm
import torch
import torch_optimizer
import torchmetrics
import torchvision.ops.focal_loss
from torch import nn, optim

from wild_boar_detection.utils import Hyperparameters


class Model(lightning.LightningModule):
    """Initializes the Autoencoder model with the specified hyperparameters.

    Args:
        hyperparameters: The hyperparameters of the autoencoder.
        loss: The loss function used for training the autoencoder.
    """

    def __init__(self: Self, hyperparameters: Hyperparameters) -> None:
        """Initializes the Autoencoder model.

        Args:
            model_name: The model to use.
            hyperparameters: The hyperparameters of the autoencoder.
            loss: The loss function used for training the autoencoder.

        Returns:
            None
        """
        super().__init__()
        self.hyperparameters: Hyperparameters = hyperparameters
        self.model = self._initialize_model(hyperparameters.MODEL)
        self.loss = self._initialize_loss(hyperparameters.LOSS)
        self.loss_weights_parameter: bool = self._initialize_loss_weights()
        self.training_step_outputs = [None]
        self.validation_step_outputs = [None]
        self.example_input_array = torch.randn(
            (1, hyperparameters.BASE_CHANNEL_SIZE, hyperparameters.INPUT_SIZE, hyperparameters.INPUT_SIZE)
        )
        # self.train_accuracy: torchmetrics.Metric = torchmetrics.classification.accuracy.BinaryAccuracy().to(
        #     device=self.device
        # )
        # self.valid_accuracy: torchmetrics.Metric = torchmetrics.classification.accuracy.BinaryAccuracy().to(
        #     device=self.device
        # )
        #
        # self.train_f1: torchmetrics.Metric = torchmetrics.classification.f_beta.BinaryF1Score().to(device=self.device)
        # self.valid_f1: torchmetrics.Metric = torchmetrics.classification.f_beta.BinaryF1Score().to(device=self.device)
        #
        # self.train_precision: torchmetrics.Metric = torchmetrics.classification.precision_recall.BinaryPrecision().to(
        #     device=self.device
        # )
        # self.valid_precision: torchmetrics.Metric = torchmetrics.classification.precision_recall.BinaryPrecision().to(
        #     device=self.device
        # )
        #
        # self.train_recall: torchmetrics.Metric = torchmetrics.classification.precision_recall.BinaryRecall().to(
        #     device=self.device
        # )
        # self.valid_recall: torchmetrics.Metric = torchmetrics.classification.precision_recall.BinaryRecall().to(
        #     device=self.device
        # )
        #
        # self.train_mcc: torchmetrics.Metric = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef().to(
        #     device=self.device
        # )
        # self.valid_mcc: torchmetrics.Metric = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef().to(
        #     device=self.device
        # )
        #
        # self.train_specificity: torchmetrics.Metric = torchmetrics.classification.specificity.BinarySpecificity().to(
        #     device=self.device
        # )
        # self.valid_specificity: torchmetrics.Metric = torchmetrics.classification.specificity.BinarySpecificity().to(
        #     device=self.device
        # )

        self.train_roc: torchmetrics.Metric = torchmetrics.classification.auroc.BinaryAUROC().to(device=self.device)
        self.valid_roc: torchmetrics.Metric = torchmetrics.classification.auroc.BinaryAUROC().to(device=self.device)

        self.train_calibration_error: torchmetrics.Metric = torchmetrics.classification.BinaryCalibrationError(
            n_bins=2, norm="l1"
        )
        self.valid_calibration_error: torchmetrics.Metric = torchmetrics.classification.BinaryCalibrationError(
            n_bins=2, norm="l1"
        )

        self.train_average_precision: torchmetrics.Metric = torchmetrics.classification.BinaryAveragePrecision()
        self.valid_average_precision: torchmetrics.Metric = torchmetrics.classification.BinaryAveragePrecision()

    @staticmethod
    def _initialize_loss(loss_name: str) -> Callable:
        if loss_name not in {"bce", "focal"}:
            raise ValueError(
                f"Loss not in list of supported losses. Supported losses: 'bce', 'focal'. Input given: {loss_name}"
            )

        if loss_name == "bce":
            return nn.functional.binary_cross_entropy_with_logits

        return torchvision.ops.focal_loss.sigmoid_focal_loss

    def _initialize_loss_weights(self: Self) -> bool:
        loss_parameters: list[str] = inspect.getfullargspec(self.loss).args
        return bool("weight" in loss_parameters or "pos_weight" in loss_parameters)

    @staticmethod
    def _initialize_model(model_name: str) -> nn.Module:
        if model_name not in {
            "mobilenetv2_035",  # ~400_00
            "mobilenetv3_small_050",
            "mobilenetv2_050",
            "mobilevit_xxs",
            "mobilenetv3_small_075",  # ~1_000_00
            "mobilevitv2_050",
            "mobilenetv2_075",
            "mobilenetv3_small_100",
            "mobilevit_xs",  # ~2_000_00
            "efficientvit_b0",
            "efficientvit_m0",
            "mobilenetv2_100",
            "mobilevitv2_075",
            "mobilenetv3_large_075",
            "efficientvit_m1",
            "mobilenetv2_110d",  # ~3_000_00
            "efficientformerv2_s0",
            "efficientnet_lite0",
            "mobileone_s1",
            "efficientvit_m2",
            "efficientnet_b0_gn",  # ~4_000_00
            "efficientnet_b0",
            "efficientnet_lite1",
            "efficientnet_es_pruned",
            "efficientnet_es",
            "mobilenetv3_rw",
            "mobilenetv3_large_100",
            "mobileone_s0",
            "mobilenetv2_140",
            "mobilevitv2_100",
            "mobilenetv2_120d",
            "efficientnet_lite2",
            "resnet10t",
            "mobilevit_s",  # ~5_000_000
        }:
            raise ValueError("Model not in list of supported models")
        try:
            return timm.create_model(model_name, pretrained=True, num_classes=1, drop_rate=0.3, drop_path_rate=0.0)
        except Exception as e:
            logging.error(e)
            try:
                return timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=0.3, drop_path_rate=0.0)
            except Exception as e:
                logging.error(e)
                return timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=0.3)

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
        if self.hyperparameters.OPTIMIZER == "adam":
            optimizer: optim.Optimizer = torch.optim.Adam(
                params=self.model.parameters(), amsgrad=True, weight_decay=self.hyperparameters.LEARNING_RATE_DECAY
            )
        elif self.hyperparameters.OPTIMIZER == "yogi":
            optimizer: optim.Optimizer = torch_optimizer.Yogi(
                params=self.model.parameters(), weight_decay=self.hyperparameters.LEARNING_RATE_DECAY
            )
        elif self.hyperparameters.OPTIMIZER == "sgd":
            optimizer: optim.Optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.hyperparameters.LEARNING_RATE,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hyperparameters.LEARNING_RATE_DECAY,
            )
        elif self.hyperparameters.OPTIMIZER == "adabelief":
            optimizer: optim.Optimizer = torch_optimizer.AdaBelief(
                params=self.model.parameters(),
                lr=self.hyperparameters.LEARNING_RATE,
                weight_decay=self.hyperparameters.LEARNING_RATE_DECAY,
                rectify=True,
                amsgrad=True,
            )
        elif self.hyperparameters.OPTIMIZER == "adahessian":
            optimizer: optim.Optimizer = torch_optimizer.Adahessian(
                self.model.parameters(),
                lr=self.hyperparameters.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=self.hyperparameters.LEARNING_RATE_DECAY,
                hessian_power=1.0,
            )
        elif self.hyperparameters.OPTIMIZER == "diffgrad":
            optimizer: optim.Optimizer = torch_optimizer.DiffGrad(
                self.model.parameters(),
                lr=self.hyperparameters.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.hyperparameters.LEARNING_RATE_DECAY,
            )
        elif self.hyperparameters.OPTIMIZER == "ranger":
            optimizer: optim.Optimizer = torch_optimizer.Ranger(
                self.model.parameters(),
                lr=self.hyperparameters.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.hyperparameters.LEARNING_RATE_DECAY,
            )
        else:
            logging.error("Unknown optimizer. Setting Adam as optimizer")
            optimizer: optim.Optimizer = torch.optim.Adam(
                params=self.model.parameters(), amsgrad=True, weight_decay=self.hyperparameters.LEARNING_RATE_DECAY
            )

        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.hyperparameters.T_0, T_mult=self.hyperparameters.T_MULT
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}

    def _log_metrics(self: Self, loss: float, y_true: torch.Tensor, y_pred: torch.Tensor, step: str = "train") -> None:
        """Logs the specified loss metric during training or validation.

        Args:
            loss: The value of the loss metric to be logged.
            step: The step or phase of the training or validation process. Defaults to "train".
        """
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        if step == "train":
            # self.train_accuracy(y_pred, y_true)
            # self.train_precision(y_pred, y_true)
            # self.train_recall(y_pred, y_true)
            # self.train_f1(y_pred, y_true)
            # self.train_mcc(y_pred, y_true)
            # self.train_specificity(y_pred, y_true)
            self.train_roc(y_pred, y_true)
            self.train_calibration_error(y_pred, y_true)
            self.train_average_precision(y_pred, y_true)

            self.log_dict(
                {
                    # f"{step}_accuracy": self.train_accuracy,
                    # f"{step}_recall": self.train_recall,
                    # f"{step}_precision": self.train_precision,
                    # f"{step}_f1": self.train_f1,
                    # f"{step}_mcc": self.train_mcc,
                    # f"{step}_specificity": self.train_specificity,
                    f"{step}_roc": self.train_roc,
                    f"{step}_calibration_error": self.train_calibration_error,
                    f"{step}_average_precision": self.train_average_precision,
                },
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=False,
            )
            return

        # self.valid_accuracy(y_pred, y_true)
        # self.valid_precision(y_pred, y_true)
        # self.valid_recall(y_pred, y_true)
        # self.valid_f1(y_pred, y_true)
        # self.valid_mcc(y_pred, y_true)
        # self.valid_specificity(y_pred, y_true)
        self.valid_roc(y_pred, y_true)
        self.valid_calibration_error(y_pred, y_true)
        self.valid_average_precision(y_pred, y_true)

        self.log_dict(
            {
                # f"{step}_accuracy": self.valid_accuracy,
                # f"{step}_recall": self.valid_recall,
                # f"{step}_precision": self.valid_precision,
                # f"{step}_f1": self.valid_f1,
                # f"{step}_mcc": self.valid_mcc,
                # f"{step}_specificity": self.valid_specificity,
                f"{step}_roc": self.valid_roc,
                f"{step}_calibration_error": self.valid_calibration_error,
                f"{step}_average_precision": self.valid_average_precision,
            },
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return

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
        x, y, pos_weight = batch
        y_pred = self.forward(x).squeeze(dim=1)

        if step == "train":
            self.training_step_outputs[0] = y_pred
        else:
            self.validation_step_outputs[0] = y_pred

        # logging.info(f"{y.shape}, {y_pred.shape}, {pos_weight.shape}")
        #
        # logging.info(
        #     torch.cat(
        #         [
        #             y.unsqueeze(dim=1),
        #             y_pred.unsqueeze(dim=1),
        #             torch.sigmoid(y_pred.unsqueeze(dim=1)),
        #             pos_weight.unsqueeze(dim=1),
        #         ],
        #         dim=-1,
        #     )
        # )
        if self.loss_weights_parameter:
            loss = self.loss(y_pred, y.float(), pos_weight=pos_weight.float(), reduction="mean")
        else:
            loss = self.loss(y_pred, y.float(), reduction="mean")

        self._log_metrics(loss=loss, step=step, y_true=y, y_pred=y_pred)
        return loss
