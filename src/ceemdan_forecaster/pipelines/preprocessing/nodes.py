# -*- coding: utf-8 -*-
"""Preprocessing (normalization) nodes.

This module implements per-IMF zero-mean normalization with strict
train/test separation. Each IMF gets its own normalizer fitted on
training data ONLY.

CRITICAL: Normalization parameters (mean, std) are computed from
training data only to prevent data leakage.
"""
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class IMFNormalizer:
    """Zero-mean normalizer for a single IMF component.

    Normalizes data using: x* = (x - mean) / std
    Denormalizes using: x = x* * std + mean

    Parameters are fit on training data only.
    """

    def __init__(self, component_name: str = ""):
        """Initialize normalizer.

        Args:
            component_name: Name of the component (for logging)
        """
        self.component_name = component_name
        self.mean = None
        self.std = None
        self._is_fitted = False

    def fit(self, train_data: np.ndarray) -> "IMFNormalizer":
        """Fit normalizer on training data.

        Args:
            train_data: Training data (1D array)

        Returns:
            Self for chaining
        """
        self.mean = float(np.mean(train_data))
        self.std = float(np.std(train_data))

        # Handle zero std (constant signal)
        if self.std == 0:
            logger.warning(f"{self.component_name}: std=0, setting to 1.0")
            self.std = 1.0

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters.

        Args:
            data: Data to normalize (1D array)

        Returns:
            Normalized data
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize data.

        Args:
            normalized: Normalized data (1D array)

        Returns:
            Original scale data
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return normalized * self.std + self.mean

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(train_data)
        return self.transform(train_data)

    @property
    def params(self) -> Dict[str, float]:
        """Get normalizer parameters."""
        return {
            "mean": self.mean,
            "std": self.std,
            "component": self.component_name,
        }


def fit_normalizers(
    val_test_decomposition: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit normalizers on training IMFs only.

    Creates one normalizer per IMF + one for residual, all fitted on
    training data only to prevent data leakage.

    Args:
        val_test_decomposition: Dictionary from transform_val_test containing:
            - train_imfs: np.ndarray of shape (n_imfs, n_train)
            - train_residual: np.ndarray of shape (n_train,)
            - n_imfs: Number of IMF components
            - ... (other fields passed through)

    Returns:
        Dictionary containing:
            - normalizers: List of fitted IMFNormalizer objects (n_imfs + 1)
            - train_imfs_norm: Normalized training IMFs
            - train_residual_norm: Normalized training residual
            - normalizer_params: List of dicts with mean/std for each component
    """
    train_imfs = val_test_decomposition["train_imfs"]
    train_residual = val_test_decomposition["train_residual"]
    n_imfs = val_test_decomposition["n_imfs"]

    logger.info("=" * 60)
    logger.info("PREPROCESSING - Fit Normalizers on Training Data")
    logger.info("=" * 60)
    logger.info(f"Number of components: {n_imfs} IMFs + 1 Residual")

    normalizers = []
    normalizer_params = []
    train_imfs_norm = np.zeros_like(train_imfs)

    # Fit normalizer for each IMF
    for i in range(n_imfs):
        component_name = f"IMF{i+1}"
        normalizer = IMFNormalizer(component_name)
        train_imfs_norm[i] = normalizer.fit_transform(train_imfs[i])
        normalizers.append(normalizer)
        normalizer_params.append(normalizer.params)
        logger.info(f"  {component_name}: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")

    # Fit normalizer for residual
    res_normalizer = IMFNormalizer("RES")
    train_residual_norm = res_normalizer.fit_transform(train_residual)
    normalizers.append(res_normalizer)
    normalizer_params.append(res_normalizer.params)
    logger.info(f"  RES: mean={res_normalizer.mean:.6f}, std={res_normalizer.std:.6f}")

    logger.info(f"Fitted {len(normalizers)} normalizers on training data")

    return {
        "normalizers": normalizers,
        "train_imfs_norm": train_imfs_norm,
        "train_residual_norm": train_residual_norm,
        "normalizer_params": normalizer_params,
        "n_imfs": n_imfs,
    }


def transform_all_data(
    val_test_decomposition: Dict[str, Any],
    normalizers_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Transform all data using fitted normalizers.

    Applies the normalizers (fitted on training data) to validation
    and test data.

    Args:
        val_test_decomposition: Dictionary from transform_val_test
        normalizers_data: Dictionary from fit_normalizers containing:
            - normalizers: List of fitted IMFNormalizer objects
            - train_imfs_norm, train_residual_norm: Already normalized train data

    Returns:
        Dictionary containing normalized data for all splits:
            - train_imfs_norm, train_residual_norm, train_dates
            - val_imfs_norm, val_residual_norm, val_dates
            - test_imfs_norm, test_residual_norm, test_dates
            - normalizers: List of fitted normalizers
            - normalizer_params: Parameters for each normalizer
            - n_imfs: Number of IMF components
    """
    normalizers = normalizers_data["normalizers"]
    n_imfs = normalizers_data["n_imfs"]

    val_imfs = val_test_decomposition["val_imfs"]
    val_residual = val_test_decomposition["val_residual"]
    test_imfs = val_test_decomposition["test_imfs"]
    test_residual = val_test_decomposition["test_residual"]

    logger.info("=" * 60)
    logger.info("PREPROCESSING - Transform Val & Test Data")
    logger.info("=" * 60)

    # Transform validation IMFs
    val_imfs_norm = np.zeros_like(val_imfs)
    for i in range(n_imfs):
        val_imfs_norm[i] = normalizers[i].transform(val_imfs[i])

    # Transform validation residual
    val_residual_norm = normalizers[-1].transform(val_residual)

    logger.info(f"Transformed validation data: {val_imfs_norm.shape[1]} samples")

    # Transform test IMFs
    test_imfs_norm = np.zeros_like(test_imfs)
    for i in range(n_imfs):
        test_imfs_norm[i] = normalizers[i].transform(test_imfs[i])

    # Transform test residual
    test_residual_norm = normalizers[-1].transform(test_residual)

    logger.info(f"Transformed test data: {test_imfs_norm.shape[1]} samples")

    # Verify normalization (train should have mean~0, std~1)
    logger.info("Verification (training data):")
    for i in range(min(3, n_imfs)):  # Show first 3 IMFs
        train_mean = np.mean(normalizers_data["train_imfs_norm"][i])
        train_std = np.std(normalizers_data["train_imfs_norm"][i])
        logger.info(f"  IMF{i+1}: mean={train_mean:.6f}, std={train_std:.6f}")

    return {
        # Training (normalized)
        "train_imfs_norm": normalizers_data["train_imfs_norm"],
        "train_residual_norm": normalizers_data["train_residual_norm"],
        "train_dates": val_test_decomposition["train_dates"],
        # Validation (normalized)
        "val_imfs_norm": val_imfs_norm,
        "val_residual_norm": val_residual_norm,
        "val_dates": val_test_decomposition["val_dates"],
        # Test (normalized)
        "test_imfs_norm": test_imfs_norm,
        "test_residual_norm": test_residual_norm,
        "test_dates": val_test_decomposition["test_dates"],
        # Normalizers (for inverse transform during inference)
        "normalizers": normalizers,
        "normalizer_params": normalizers_data["normalizer_params"],
        "n_imfs": n_imfs,
    }
