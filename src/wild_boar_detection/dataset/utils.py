from __future__ import annotations

import typing
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

if typing.TYPE_CHECKING:
    import pandas as pd


def get_splits(
    data: pd.DataFrame | np.ndarray | list[...],
    train_size: float,
    valid_size: float,
    test_size: float,
    stratify_col: str | None = None,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Splits the data into train, validation, and test sets.

    Args:
        data: The input data to be split. It can be a pandas DataFrame, numpy array, or list.
        train_size: The proportion of the data to be used for training.
        valid_size: The proportion of the data to be used for validation.
        test_size: The proportion of the data to be used for testing.
        stratify_col: The column name used for stratified sampling. Defaults to None.
        seed: The random seed for reproducibility. Defaults to 42.

    Returns:
        A tuple containing the train, validation, and test splits of the data.

    Raises:
        AssertionError: If the sum of train_size, valid_size, and test_size is greater than 1.

    Examples:
        >>> data = pd.DataFrame({'feature': [1, 2, 3, 4, 5], 'label': [0, 1, 0, 1, 0]})
        >>> train, valid, test = get_splits(data, train_size=0.6, valid_size=0.2, test_size=0.2, stratify_col='label')
    """
    assert train_size + valid_size + test_size <= 1.0

    if stratify_col:
        train_split, valid_test = train_test_split(
            data, train_size=train_size, stratify=data[stratify_col], random_state=seed
        )
        valid_split, test_split = train_test_split(
            valid_test, train_size=valid_size / (1 - train_size), stratify=valid_test[stratify_col], random_state=seed
        )
    else:
        train_split, valid_test = train_test_split(data, train_size=train_size, stratify=None, random_state=seed)
        valid_split, test_split = train_test_split(
            valid_test, train_size=valid_size / (1 - train_size), stratify=None, random_state=seed
        )

    return train_split, valid_split, test_split
