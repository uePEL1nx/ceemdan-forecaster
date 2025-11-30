# -*- coding: utf-8 -*-
"""Data loading nodes.

This module implements the data loading functionality for the
CEEMDAN-Informer-LSTM pipeline, loading time series data from CSV files.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_data(data_source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load time series data from CSV file.

    This function replicates the data loading logic from the Panel pipeline's
    DataLoadingStage, extracting price data and dates from a CSV file.

    Args:
        data_source: Dictionary with data source configuration:
            - path: Path to CSV file
            - date_column: Name of date column
            - value_column: Name of value column (typically Close or Price)
            - open_column: Name of open price column (optional, for backtest)
            - encoding: File encoding (default: utf-8)

    Returns:
        Dictionary containing:
            - data: np.ndarray of float64 values (the time series)
            - dates: List of datetime objects
            - open_prices: np.ndarray of open prices (None if not available)
            - n_samples: Number of data points
            - source: Source file path

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If specified columns are not found in the CSV
    """
    path = Path(data_source["path"])
    date_column = data_source["date_column"]
    value_column = data_source["value_column"]
    open_column = data_source.get("open_column", "Open")
    encoding = data_source.get("encoding", "utf-8")

    logger.info(f"Loading data from: {path}")

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load CSV
    df = pd.read_csv(path, encoding=encoding)
    logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

    # Validate columns exist
    if value_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise KeyError(
            f"Value column '{value_column}' not found. Available columns: {available}"
        )

    if date_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise KeyError(
            f"Date column '{date_column}' not found. Available columns: {available}"
        )

    # Extract data - match Panel pipeline's exact approach
    data = df[value_column].values.astype(np.float64)
    dates = list(pd.to_datetime(df[date_column]))

    # Extract Open prices if available (for backtest execution timing)
    open_prices = None
    if open_column in df.columns:
        open_prices = df[open_column].values.astype(np.float64)
        logger.info(f"Open prices loaded from column '{open_column}'")
    else:
        logger.info(f"Open column '{open_column}' not found - backtest will use Close prices only")

    # Validate data
    n_samples = len(data)
    if n_samples == 0:
        raise ValueError("CSV file contains no data")

    # Check for NaN values
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        logger.warning(f"Data contains {nan_count} NaN values ({nan_count/n_samples*100:.2f}%)")

    # Log summary statistics
    logger.info(f"Data loaded successfully:")
    logger.info(f"  Samples: {n_samples:,}")
    logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"  Value range: {data.min():.4f} to {data.max():.4f}")
    logger.info(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")

    return {
        "data": data,
        "dates": dates,
        "open_prices": open_prices,
        "n_samples": n_samples,
        "source": str(path),
    }
