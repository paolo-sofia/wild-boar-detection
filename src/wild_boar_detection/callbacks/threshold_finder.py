import logging
from typing import Any, Self

import lightning.pytorch as pl
import torch
import torchmetrics
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

logger = logging.getLogger(__name__)


class ThresholdFinder(Callback):
    def __init__(
        self: Self, metric: torchmetrics.Metric, task: str = "binary", num_classes: int = 1, mode: str = "max"
    ) -> None:
        super().__init__()
        self.train_metric: dict[float, torchmetrics.Metric] = None
        self.validation_metric: dict[float, torchmetrics.Metric] = None
        self.mode = mode
        self.train_threshold: float = 0.0
        self.validation_threshold: float = 0.0
        self.__init_metrics(metric=metric, task=task, num_classes=num_classes, mode=mode)

    def __init_metrics(self: Self, metric: torchmetrics.Metric, task: str, mode: str, num_classes: int) -> None:
        if mode == "max":
            thresholds = torch.arange(start=1.0, end=0.0, step=-0.05, dtype=torch.float32)
        else:
            thresholds = torch.arange(start=0.0, end=1.0, step=0.05, dtype=torch.float32)

        self.train_metric: dict[float, torchmetrics.Metric] = {}
        self.validation_metric: dict[float, torchmetrics.Metric] = {}

        for threshold in thresholds:
            th: float = threshold.item()
            self.train_metric[th] = metric(task=task, threshold=th, num_classes=num_classes)
            self.validation_metric[th] = metric(task=task, threshold=th, num_classes=num_classes)

    def _update_metrics(
        self,
        pl_module: "pl.LightningModule",
        batch: Any,
        metrics_dict: dict[float, torchmetrics.Metric],
        predictions: torch.Tensor | None,
    ) -> None:
        x, y, weights = batch
        if predictions is None:
            return

        for metric in metrics_dict.values():
            metric.to(pl_module.device).update(predictions, y)

    def _find_threshold(self: Self, metrics_dict: dict[float, torchmetrics.Metric]) -> tuple[float, float]:
        best_score = 0.0 if self.mode == "max" else 1.0
        best_threshold: float = -1.0

        for threshold, metric in metrics_dict.items():
            metric_value = metric.compute()
            metric.reset()

            if self.mode == "max" and metric_value > best_score or self.mode == "min" and metric_value < best_score:
                best_score = metric_value
                best_threshold = threshold

        return best_threshold, best_score

    @override
    def on_train_batch_end(
        self: Self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # logger.error("BATCH END OF TRAINING")
        self._update_metrics(
            pl_module=pl_module,
            batch=batch,
            metrics_dict=self.train_metric,
            predictions=pl_module.training_step_outputs[0],
        )

    @override
    def on_train_epoch_end(self: Self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        threshold, score = self._find_threshold(self.train_metric)
        pl_module.log_dict(
            {"train_threshold": threshold, "train_score": score},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    @override
    def on_validation_batch_end(
        self: Self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # logger.error("BATCH END OF VALIDATION")
        self._update_metrics(
            pl_module=pl_module,
            batch=batch,
            metrics_dict=self.validation_metric,
            predictions=pl_module.validation_step_outputs[0],
        )

    @override
    def on_validation_epoch_end(self: Self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        threshold, score = self._find_threshold(self.validation_metric)

        pl_module.log_dict(
            {"valid_threshold": threshold, "valid_score": score},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
