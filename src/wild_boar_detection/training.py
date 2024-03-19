#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pathlib
import sys

import lightning
import tomllib
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(str(pathlib.Path.cwd()))

from src.model.autoencoder import Autoencoder, initialize_weights
from src.model.dataset.audio_dataset import AudioDataset
from src.model.dataset.utils import get_splits
from src.utils import Hyperparameters, dataclass_from_dict

torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(prog="Model training", description="Trains the autoencoder")
parser.add_argument("experiment_name", type=str)
cli_args: argparse.Namespace = parser.parse_args()
EXPERIMENT_NAME: str = cli_args.experiment_name


def get_dataloaders(
    train: list, valid: list, test: list, mel_spectrogram_params: dict[str, int]
) -> dict[str, torch.utils.data.DataLoader]:
    dataloaders: dict[str, torch.utils.data.DataLoader] = {"train": None, "valid": None, "test": None}
    for split_name, split in {"train": train, "valid": valid, "test": test}.items():
        dataloaders[split_name] = DataLoader(
            dataset=AudioDataset(
                data_path=split,
                mode=split_name,
                crop_size=cfg.CROP_SIZE_SECONDS,
                precision=cfg.PRECISION,
                mel_spectrogram_param=mel_spectrogram_params,
            ),
            batch_size=cfg.BATCH_SIZE,
            num_workers=0,
            shuffle=split_name == "train",
            pin_memory=True,
            persistent_workers=False,
        )

    return dataloaders


with (pathlib.Path.cwd() / "config" / "model.toml").open("rb") as f:
    cfg: Hyperparameters | dict[str, int | float | str | bool] = dataclass_from_dict(Hyperparameters, tomllib.load(f))

lightning.seed_everything(cfg.SEED, workers=True)

songs_path: list[pathlib.Path] = list(pathlib.Path(pathlib.Path.cwd()).parent.rglob("*.safetensors"))
train, valid, test = get_splits(
    data=songs_path, train_size=cfg.TRAIN_SIZE, valid_size=cfg.VALID_SIZE, test_size=cfg.TEST_SIZE, stratify_col=None
)

mel_spectrogram_params: dict[str, int] = {
    "N_FFT": cfg.N_FFT,
    "N_MELS": cfg.N_MELS,
    "WIN_LENGTH": cfg.WIN_LENGTH,
    "HOP_LENGTH": cfg.HOP_LENGTH,
}
dataloaders = get_dataloaders(train=train, valid=valid, test=test, mel_spectrogram_params=mel_spectrogram_params)

model = Autoencoder(hyperparam=cfg)
model.apply(initialize_weights)
model._log_hyperparams = False

savedir: pathlib.Path = pathlib.Path("/home/paolo/git/spotify-playlist-generator/logs/mlruns")
# experiment_name: str = "No Duplicates - No resize - No regularization"
logger = MLFlowLogger(
    experiment_name="Autoencoder",
    save_dir=str(savedir),
    log_model=True,
    run_name=f"{EXPERIMENT_NAME} - Overfit Batch" if cfg.OVERFIT_BATCHES else EXPERIMENT_NAME,
)

early_stop_callback: EarlyStopping = EarlyStopping(
    monitor="valid_loss",
    min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
    patience=cfg.EARLY_STOPPING_PATIENCE,
    verbose=True,
    mode="min",
    check_finite=True,
)

model_checkpoint: ModelCheckpoint = ModelCheckpoint(
    dirpath=str(
        savedir / logger.experiment_id / logger.run_id / "artifacts" / "model" / "checkpoints" / "model_checkpoint"
    ),
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

callbacks: list = [RichProgressBar()] if cfg.OVERFIT_BATCHES else [early_stop_callback, RichProgressBar()]
callbacks.append(model_checkpoint)

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
    gradient_clip_val=cfg.GRADIENT_CLIP_VAL,
    gradient_clip_algorithm=cfg.GRADIENT_CLIP_TYPE,
    accumulate_grad_batches=cfg.GRADIENT_ACCUMULATION_BATCHES,
)

logger.log_hyperparams(cfg.__dict__)
# print(cfg.__dict__)
trainer.fit(
    model=model, train_dataloaders=dataloaders.get("train"), val_dataloaders=dataloaders.get("valid"), ckpt_path=None
)
trainer.test(model=model, dataloaders=dataloaders.get("test"), ckpt_path="best")
os.system("notify-send 'Training complete!'")
