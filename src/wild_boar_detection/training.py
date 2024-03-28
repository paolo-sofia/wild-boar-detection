#!/usr/bin/env python
# coding: utf-8
import json
import logging
import os
import pathlib
import sys

import lightning
import torch
import torchmetrics.classification
from dotenv import dotenv_values
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from wild_boar_detection.callbacks.threshold_finder import ThresholdFinder
from wild_boar_detection.dataset.image_dataset import ImageDataset
from wild_boar_detection.model import Model
from wild_boar_detection.utils import Hyperparameters, dataclass_from_dict

sys.path.append(str(pathlib.Path.cwd()))
torch.set_float32_matmul_precision("medium")
cfg: Hyperparameters | dict[str, int | float | str | bool] = dataclass_from_dict(Hyperparameters, dotenv_values())
lightning.seed_everything(cfg.SEED, workers=True)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPLETED_RUNS_PATH: pathlib.Path = pathlib.Path("../../data/completed_runs.json")


def log_best_epoch_metric(logger: MLFlowLogger, metrics_path: pathlib.Path) -> None:
    """Logs the best epoch metrics to the specified metrics path.

    Args:
        logger: The MLFlowLogger instance used for logging.
        metrics_path: The path to the metrics directory.

    Returns:
        None
    """
    print(f"Logging best epoch metrics to {metrics_path}")
    with metrics_path.joinpath("valid_loss").open("r") as f:
        valid_loss: list[tuple[int, float]] = list(enumerate(float(x.split(" ")[1]) for x in f.readlines()))

    best_index: int
    best_loss: float
    best_index, best_loss = min(valid_loss, key=lambda x: x[1])

    metrics_dict: dict[str, float] = {"loss": best_loss}

    for metric_path in metrics_path.rglob("valid*"):
        if metric_path.name.endswith("loss"):
            continue
        metric_name: str = "_".join(metric_path.name.split("_")[1:])

        with metric_path.open("r") as f:
            metric: list[float] = [float(x.split(" ")[1]) for x in f.readlines()]

            metrics_dict[metric_name] = metric[best_index]

    logger.log_metrics(metrics_dict)


def main(model_list: list[str], loss_list: list[str], optimizer_list: list[str]) -> None:
    """Trains a model using the specified dataloaders and hyperparameters.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    try:
        with pathlib.Path(COMPLETED_RUNS_PATH).open("r") as f:
            completed_runs: list[str] = json.load(f)
    except Exception as e:
        logging.error(e)
        completed_runs = []

    try:
        for model_name in model_list:
            for loss in loss_list:
                for optimizer in optimizer_list:
                    torch.cuda.empty_cache()

                    cfg.OPTIMIZER = optimizer
                    cfg.MODEL = model_name
                    cfg.LOSS = loss

                    run_name: str = f"{cfg.MODEL}_{cfg.LOSS}_{cfg.OPTIMIZER}"
                    run_name = f"{run_name} - Overfit Batch" if cfg.OVERFIT_BATCHES else run_name

                    if run_name in completed_runs:
                        logging.warning(f"Run {run_name} already completed, skipping")
                        continue

                    savedir: pathlib.Path = pathlib.Path("../../data/logs/mlruns")
                    # experiment_name: str = "No Duplicates - No resize - No regularization"
                    early_stop_callback: EarlyStopping = EarlyStopping(
                        monitor="valid_loss",
                        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
                        patience=cfg.EARLY_STOPPING_PATIENCE,
                        verbose=True,
                        mode="min",
                        check_finite=True,
                    )

                    logger: MLFlowLogger = MLFlowLogger(
                        experiment_name="Model training",
                        save_dir=str(savedir),
                        log_model=True,
                        run_name=run_name,
                    )

                    MODEL_OUTPUT_DIR: pathlib.Path = (
                        savedir / logger.experiment_id / logger.run_id / "artifacts" / "model"
                    )

                    model_checkpoint: ModelCheckpoint = ModelCheckpoint(
                        dirpath=str(MODEL_OUTPUT_DIR / "checkpoints" / "model_checkpoint"),
                        filename="model_checkpoint",
                        monitor="valid_loss",
                        verbose=True,
                        save_last=None,
                        save_top_k=1,
                        save_weights_only=False,
                        mode="min",
                        auto_insert_metric_name=True,
                        every_n_train_steps=None,
                        train_time_interval=None,
                        every_n_epochs=None,
                        save_on_train_epoch_end=None,
                        enable_version_counter=True,
                    )

                    train_dataloader: DataLoader = DataLoader(
                        dataset=ImageDataset(
                            data_path=pathlib.Path("../../data/train.parquet").resolve(), mode="train"
                        ),
                        batch_size=cfg.BATCH_SIZE,
                        num_workers=cfg.NUM_WORKERS,
                        shuffle=True,
                        pin_memory=True,
                        persistent_workers=False,
                    )

                    valid_dataloader: DataLoader = DataLoader(
                        dataset=ImageDataset(
                            data_path=pathlib.Path("../../data/valid.parquet").resolve(), mode="valid"
                        ),
                        batch_size=cfg.BATCH_SIZE,
                        num_workers=cfg.NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=False,
                    )

                    try:
                        model: Model = Model(hyperparameters=cfg)
                        # model.apply(initialize_weights)
                        model._log_hyperparams = False

                        callbacks: list = (
                            [RichProgressBar()] if cfg.OVERFIT_BATCHES else [early_stop_callback, RichProgressBar()]
                        )
                        callbacks.append(model_checkpoint)
                        callbacks.append(
                            ThresholdFinder(
                                metrics=[
                                    torchmetrics.classification.accuracy.Accuracy,
                                    torchmetrics.classification.f_beta.F1Score,
                                    torchmetrics.classification.precision_recall.Precision,
                                    torchmetrics.classification.precision_recall.Recall,
                                    torchmetrics.classification.matthews_corrcoef.MatthewsCorrCoef,
                                    torchmetrics.classification.specificity.Specificity,
                                ]
                            )
                        )
                        trainer: lightning.Trainer = lightning.Trainer(
                            accelerator="gpu",
                            num_nodes=1,
                            precision=cfg.PRECISION,
                            logger=logger,
                            callbacks=callbacks,
                            fast_dev_run=False,
                            max_epochs=cfg.EPOCHS,
                            min_epochs=1,
                            overfit_batches=cfg.OVERFIT_BATCHES,
                            log_every_n_steps=100,
                            check_val_every_n_epoch=1,
                            enable_checkpointing=True,
                            enable_progress_bar=True,
                            enable_model_summary=True,
                            deterministic="warn",
                            benchmark=True,
                            inference_mode=True,
                            profiler=None,  # AdvancedProfiler(),
                            detect_anomaly=False,
                            barebones=False,
                            gradient_clip_val=None,  # cfg.GRADIENT_CLIP_VAL,
                            gradient_clip_algorithm=cfg.GRADIENT_CLIP_TYPE,
                            accumulate_grad_batches=cfg.GRADIENT_ACCUMULATION_BATCHES,
                        )

                        logger.log_hyperparams(cfg.__dict__)
                        trainer.fit(
                            model=model,
                            train_dataloaders=train_dataloader,
                            val_dataloaders=valid_dataloader,
                            ckpt_path=None,
                        )
                        # trainer.test(model=model, dataloaders=dataloaders.get("test"), ckpt_path="best")

                        model: Model = Model.load_from_checkpoint(
                            MODEL_OUTPUT_DIR / "checkpoints" / "model_checkpoint" / "model_checkpoint.ckpt",
                            hyperparameters=cfg,
                        )

                        model.to_onnx(file_path=MODEL_OUTPUT_DIR / f"{cfg.MODEL}.onnx")

                        log_best_epoch_metric(
                            logger=logger, metrics_path=savedir / logger.experiment_id / logger.run_id / "metrics"
                        )

                        completed_runs.append(run_name)
                    except Exception as e:
                        logging.error(e)
                        completed_runs.append(run_name)

    except Exception as e:
        logging.error(e)
    finally:
        with pathlib.Path(COMPLETED_RUNS_PATH).open("w") as f:
            json.dump(completed_runs, f)
    os.system("notify-send 'Training complete!'")


if __name__ == "__main__":
    model_list: list[str] = [
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
    ]
    optimizer_list: list[str] = ["adam"]  # , "sgd", "yogi", "ranger"]
    loss_list: list[str] = ["bce"]

    main(optimizer_list=optimizer_list, loss_list=loss_list, model_list=model_list)
