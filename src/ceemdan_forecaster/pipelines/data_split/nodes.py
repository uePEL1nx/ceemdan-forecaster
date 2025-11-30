# -*- coding: utf-8 -*-
"""Data split nodes.

This module implements temporal data splitting for the CEEMDAN-Informer-LSTM
pipeline. It enforces strict chronological ordering to prevent data leakage.

CRITICAL: No shuffling is performed. Train data is always BEFORE validation,
and validation is always BEFORE test data.
"""
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def temporal_split(
    raw_data: Dict[str, Any],
    data_split: Dict[str, Any]
) -> Dict[str, Any]:
    """Split time series data temporally into train/val/test sets.

    This function implements strict temporal separation to prevent data leakage.
    The split is chronological - training data comes first, then validation,
    then test data.

    Args:
        raw_data: Dictionary from load_csv_data containing:
            - data: np.ndarray of float64 values
            - dates: List of datetime objects
            - open_prices: np.ndarray of open prices (optional, for backtest)
            - n_samples: Number of data points
            - source: Source file path
        data_split: Dictionary with split configuration:
            - train_ratio: Fraction for training (default: 0.80)
            - val_ratio: Fraction for validation (default: 0.10)
            - test_ratio: Fraction for testing (default: 0.10)
            - strict_separation: Whether to enforce strict temporal ordering

    Returns:
        Dictionary containing:
            - train_data: np.ndarray of training values
            - train_dates: List of training dates
            - val_data: np.ndarray of validation values
            - val_dates: List of validation dates
            - test_data: np.ndarray of test values
            - test_dates: List of test dates
            - test_open_prices: np.ndarray of test open prices (for backtest)
            - split_indices: Dict with train_end_idx and val_end_idx
            - n_train: Number of training samples
            - n_val: Number of validation samples
            - n_test: Number of test samples

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance)
    """
    data = raw_data["data"]
    dates = raw_data["dates"]
    n_samples = raw_data["n_samples"]
    open_prices = raw_data.get("open_prices")

    # Get split ratios
    train_ratio = data_split.get("train_ratio", 0.80)
    val_ratio = data_split.get("val_ratio", 0.10)
    test_ratio = data_split.get("test_ratio", 0.10)
    strict_separation = data_split.get("strict_separation", True)

    # Validate ratios sum to 1.0
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    # Calculate split indices (no shuffling - chronological order)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val  # Remaining samples go to test

    train_end_idx = n_train
    val_end_idx = n_train + n_val

    logger.info(f"Splitting {n_samples:,} samples with ratios: "
                f"train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

    # Perform the split
    train_data = data[:train_end_idx]
    train_dates = dates[:train_end_idx]

    val_data = data[train_end_idx:val_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]

    test_data = data[val_end_idx:]
    test_dates = dates[val_end_idx:]

    # Split open prices if available (for backtest)
    test_open_prices = None
    if open_prices is not None:
        test_open_prices = open_prices[val_end_idx:]
        logger.info(f"Open prices split for test set: {len(test_open_prices)} samples")

    # Log split information
    logger.info(f"Split completed (strict_separation={strict_separation}):")
    logger.info(f"  Train: {n_train:,} samples ({n_train/n_samples*100:.1f}%)")
    logger.info(f"    Date range: {train_dates[0]} to {train_dates[-1]}")
    logger.info(f"  Val:   {n_val:,} samples ({n_val/n_samples*100:.1f}%)")
    logger.info(f"    Date range: {val_dates[0]} to {val_dates[-1]}")
    logger.info(f"  Test:  {n_test:,} samples ({n_test/n_samples*100:.1f}%)")
    logger.info(f"    Date range: {test_dates[0]} to {test_dates[-1]}")

    # Validate no overlap (sanity check)
    if strict_separation:
        assert train_dates[-1] < val_dates[0], "Train/Val overlap detected!"
        assert val_dates[-1] < test_dates[0], "Val/Test overlap detected!"
        logger.info("Strict temporal separation verified: No overlap between splits")

    return {
        "train_data": train_data,
        "train_dates": train_dates,
        "val_data": val_data,
        "val_dates": val_dates,
        "test_data": test_data,
        "test_dates": test_dates,
        "test_open_prices": test_open_prices,
        "split_indices": {
            "train_end_idx": train_end_idx,
            "val_end_idx": val_end_idx,
        },
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
