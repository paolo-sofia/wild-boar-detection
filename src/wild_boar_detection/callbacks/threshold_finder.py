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
        self: Self, metrics: list[torchmetrics.Metric], task: str = "binary", num_classes: int = 1, mode: str = "max"
    ) -> None:
        super().__init__()
        self.train_metrics: dict[str, dict[float, torchmetrics.Metric]] = {k.__name__: {} for k in metrics}
        self.validation_metrics: dict[str, dict[float, torchmetrics.Metric]] = {k.__name__: {} for k in metrics}
        self.mode = mode
        self.__init_metrics(metrics=metrics, task=task, num_classes=num_classes, mode=mode)

    def __init_metrics(self: Self, metrics: list[torchmetrics.Metric], task: str, mode: str, num_classes: int) -> None:
        if mode == "max":
            thresholds = torch.arange(start=1.0, end=0.0, step=-0.05, dtype=torch.float32)
        else:
            thresholds = torch.arange(start=0.0, end=1.0, step=0.05, dtype=torch.float32)

        for threshold in thresholds:
            th: float = threshold.item()
            for metric in metrics:
                self.train_metrics[metric.__name__][th] = metric(task=task, threshold=th, num_classes=num_classes)
                self.validation_metrics[metric.__name__][th] = metric(task=task, threshold=th, num_classes=num_classes)

    def _update_metrics(
        self,
        pl_module: "pl.LightningModule",
        batch: Any,
        metrics_dict: dict[str, dict[float, torchmetrics.Metric]],
        predictions: torch.Tensor | None,
    ) -> None:
        x, y, weights = batch
        if predictions is None:
            return

        for metrics_at_threshold in metrics_dict.values():
            for metric in metrics_at_threshold.values():
                metric.to(pl_module.device).update(predictions, y)

    def _find_threshold(
        self: Self, metrics_dict: dict[str, dict[float, torchmetrics.Metric]]
    ) -> dict[str, tuple[float, float]]:
        metrics_results: dict[str, tuple[float, float]] = {}

        for metric_name, metric_at_thresholds in metrics_dict.items():
            best_score: float = 0.0 if self.mode == "max" else 1.0
            best_threshold: float = -1.0

            for threshold, metric in metric_at_thresholds.items():
                metric_value: float = metric.compute().item()
                metric.reset()

                if (
                    self.mode == "max"
                    and metric_value >= best_score
                    or self.mode == "min"
                    and metric_value <= best_score
                ):
                    best_score = metric_value
                    best_threshold = threshold

                # logger.error(
                #     f"Metric {metric_name} - Threshold {threshold:.2f} - Metric {metric_value:.2f} - Best Score {best_score:.2f} - Best threshold {best_threshold:.2f}"
                # )

            metrics_results[metric_name] = (round(best_threshold, 3), round(best_score, 3))
        return metrics_results

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
            metrics_dict=self.train_metrics,
            predictions=pl_module.training_step_outputs[0],
        )

    @override
    def on_train_epoch_end(self: Self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics_results: dict[str, tuple[float, float]] = self._find_threshold(self.train_metrics)

        log_metrics_results: dict[str, float] = {}
        for metric_name, (threshold, score) in metrics_results.items():
            log_metrics_results[f"train_{metric_name}_threshold"] = round(threshold, 3)
            log_metrics_results[f"train_{metric_name}"] = round(score, 3)

        pl_module.log_dict(
            dictionary=log_metrics_results,
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
            metrics_dict=self.validation_metrics,
            predictions=pl_module.validation_step_outputs[0],
        )

    @override
    def on_validation_epoch_end(self: Self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics_results: dict[str, tuple[float, float]] = self._find_threshold(self.validation_metrics)

        log_metrics_results: dict[str, float] = {}
        for metric_name, (threshold, score) in metrics_results.items():
            log_metrics_results[f"valid_{metric_name}_threshold"] = round(threshold, 3)
            log_metrics_results[f"valid_{metric_name}"] = round(score, 3)

        pl_module.log_dict(
            dictionary=log_metrics_results,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
