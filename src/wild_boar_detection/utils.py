import dataclasses
import logging
import os
import pathlib
import sys
from functools import lru_cache
from typing import Type

from torch import nn


@dataclasses.dataclass
class Hyperparameters:
    """Data class for storing hyperparameters used in training.

    Attributes:
        BATCH_SIZE (int): The batch size for training.
        INPUT_SIZE (int): The size of the input tensor.
        EPOCHS (int): The number of training epochs.
        LEARNING_RATE (float): The learning rate for the optimizer.
        LEARNING_RATE_DECAY (int): The learning rate decay.
        TRAIN_SIZE (float): The proportion of the data used for training.
        BASE_CHANNEL_SIZE (int): The base number of channels for the model.
        EARLY_STOPPING_PATIENCE (int): The patience for early stopping.
        EARLY_STOPPING_MIN_DELTA (float): The minimum delta for early stopping.
        OVERFIT_BATCHES (int): The number of batches to overfit on.
        LOSS (nn.Module): The loss function used for training.
        SEED (int): The random seed for reproducibility.
        PRECISION (int): The floating point precision used for training.
        GRADIENT_ACCUMULATION_BATCHES (int): The number of batches to run before updating the weights
    """

    BATCH_SIZE: int
    INPUT_SIZE: int
    TRAIN_SIZE: float
    BASE_CHANNEL_SIZE: int
    VALID_SIZE: float
    TEST_SIZE: float
    EPOCHS: int
    LEARNING_RATE: float
    LEARNING_RATE_DECAY: float
    EARLY_STOPPING_PATIENCE: int
    EARLY_STOPPING_MIN_DELTA: float
    OVERFIT_BATCHES: int
    SEED: int
    GRADIENT_CLIP_VAL: float
    GRADIENT_CLIP_TYPE: str
    PRECISION: int
    GRADIENT_ACCUMULATION_BATCHES: int
    LOSS: nn.Module = nn.L1Loss


@lru_cache()
def set_logger(name: str) -> logging.Logger:
    """Set up and return a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.

    Examples:
        >>> logger = set_logger("my_logger")
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def get_dir_absolute_path(dir_name: str) -> pathlib.Path:
    """Return the absolute path of the directory with the specified name.

    It searches in all subdirectories of the cwd parent folder and returns the absolute path of the directory named
    `dir_name`
    Args:
        dir_name (str): The name of the directory.

    Returns:
        pathlib.Path: The absolute path of the directory.

    Examples:
        >>> dir_path = get_dir_absolute_path("my_directory")
    """
    current_folder: pathlib.Path = pathlib.Path.cwd()

    target_folder_path: pathlib.Path = pathlib.Path()
    for parent in current_folder.parents:
        for potential_folder_path in parent.rglob(dir_name):
            if potential_folder_path.is_dir():
                return potential_folder_path

    return target_folder_path


def dataclass_from_dict(class_: Type, dictionary: dict[str, str | float | int]) -> dict[str, str | float | int] | Type:
    """Converts a dictionary to a dataclass instance.

    Args:
        class_: The dataclass type.
        dictionary: The dictionary to convert.

    Returns:
        Union[dict[str, Union[str, float, int]], dataclasses.dataclass]: The converted dataclass instance or the
            original dictionary.
    """
    try:
        field_types: dict = {f.name: f.type for f in dataclasses.fields(class_)}
        # print(f"field_types: {field_types}")
        # print(f"dictionary: {dictionary}")
        return class_(**{f: dataclass_from_dict(field_types.get(f), dictionary.get(f)) for f in dictionary})
    except Exception:
        # print(f"EXCEPTION: {e}")
        return dictionary  # The object is not a dataclass field


logger: logging.Logger = set_logger(os.getenv("TITLE", ""))