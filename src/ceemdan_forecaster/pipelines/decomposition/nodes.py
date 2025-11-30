# -*- coding: utf-8 -*-
"""CEEMDAN decomposition nodes.

This module implements CEEMDAN decomposition with strict train/test separation.
The decomposer is fit ONLY on training data to prevent data leakage.

CRITICAL: The fit/transform pattern ensures:
- Decomposer parameters learned from training data only
- Validation and test data decomposed independently
- IMF count aligned across all splits for consistent downstream processing
"""
import logging
from typing import Any, Dict

import numpy as np
from PyEMD import CEEMDAN

logger = logging.getLogger(__name__)


class CEEMDANDecomposer:
    """CEEMDAN decomposer with fit/transform pattern for Kedro.

    This is a simplified version optimized for the Kedro pipeline that:
    - Fits on training data to determine IMF count
    - Transforms validation and test data with IMF alignment
    - Stores decomposer state for serialization
    """

    def __init__(self, trials: int = 100, epsilon: float = 0.005):
        """Initialize CEEMDAN decomposer.

        Args:
            trials: Number of noise trials for CEEMDAN
            epsilon: Noise amplitude for CEEMDAN
        """
        self.trials = trials
        self.epsilon = epsilon
        self.ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)

        # Fit state
        self._is_fitted = False
        self._fitted_n_imfs = None

    def fit_decompose(self, train_data: np.ndarray) -> tuple:
        """Fit decomposer on training data and return decomposition.

        Args:
            train_data: Training time series (1D array)

        Returns:
            Tuple of (imfs, residual) where:
                - imfs: np.ndarray of shape (n_imfs, n_samples)
                - residual: np.ndarray of shape (n_samples,)
        """
        logger.info(f"Fitting CEEMDAN on training data (n={len(train_data)})...")
        logger.info(f"  Parameters: trials={self.trials}, epsilon={self.epsilon}")

        # Run CEEMDAN decomposition
        decomposition = self.ceemdan(train_data)

        # Separate IMFs from residual (last component is residual)
        imfs = decomposition[:-1]
        residual = decomposition[-1]

        self._fitted_n_imfs = len(imfs)
        self._is_fitted = True

        # Validate reconstruction
        reconstructed = np.sum(imfs, axis=0) + residual
        max_error = np.max(np.abs(train_data - reconstructed))

        logger.info(f"  Decomposition: {self._fitted_n_imfs} IMFs + 1 Residual")
        logger.info(f"  Reconstruction error: {max_error:.6e}")

        if max_error > 1e-6:
            logger.warning(f"Reconstruction error {max_error:.6e} exceeds 1e-6!")

        return imfs, residual

    def transform(self, data: np.ndarray) -> tuple:
        """Transform data using fitted decomposer.

        Performs CEEMDAN on new data and aligns IMF count to match training.

        Args:
            data: Time series to decompose (1D array)

        Returns:
            Tuple of (imfs, residual) aligned to training IMF count

        Raises:
            RuntimeError: If decomposer not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Decomposer not fitted. Call fit_decompose first.")

        logger.info(f"Transforming data (n={len(data)})...")

        # Run CEEMDAN on new data
        decomposition = self.ceemdan(data)

        imfs = decomposition[:-1]
        residual = decomposition[-1]
        n_imfs = len(imfs)

        logger.info(f"  Decomposition: {n_imfs} IMFs + 1 Residual")

        # Align IMF count if needed
        if n_imfs != self._fitted_n_imfs:
            logger.warning(f"IMF count mismatch: got {n_imfs}, expected {self._fitted_n_imfs}")
            imfs, residual = self._align_imf_count(imfs, residual, len(data))

        # Validate reconstruction
        reconstructed = np.sum(imfs, axis=0) + residual
        max_error = np.max(np.abs(data - reconstructed))
        logger.info(f"  Reconstruction error: {max_error:.6e}")

        return imfs, residual

    def _align_imf_count(self, imfs: np.ndarray, residual: np.ndarray,
                         signal_length: int) -> tuple:
        """Align IMF count to match fitted model.

        If fewer IMFs: pad with zeros
        If more IMFs: merge extras into residual
        """
        n_imfs = len(imfs)
        target = self._fitted_n_imfs

        if n_imfs < target:
            # Pad with zeros
            padding = np.zeros((target - n_imfs, signal_length))
            aligned_imfs = np.vstack([imfs, padding])
            aligned_residual = residual
            logger.info(f"  Padded {target - n_imfs} zero IMFs")

        elif n_imfs > target:
            # Merge extras into residual
            aligned_imfs = imfs[:target]
            extra_imfs = imfs[target:]
            aligned_residual = residual + np.sum(extra_imfs, axis=0)
            logger.info(f"  Merged {n_imfs - target} extra IMFs into residual")

        else:
            aligned_imfs = imfs
            aligned_residual = residual

        return aligned_imfs, aligned_residual

    @property
    def n_imfs(self) -> int:
        """Number of IMFs from fitted decomposition."""
        return self._fitted_n_imfs

    @property
    def is_fitted(self) -> bool:
        """Whether decomposer has been fitted."""
        return self._is_fitted


def fit_decompose_train(
    split_data: Dict[str, Any],
    ceemdan: Dict[str, Any]
) -> Dict[str, Any]:
    """Fit CEEMDAN decomposer on training data only.

    This node ensures CEEMDAN is fit ONLY on training data to prevent
    data leakage. The decomposer state is stored for transforming
    validation and test data.

    Args:
        split_data: Dictionary from temporal_split containing:
            - train_data: np.ndarray of training values
            - train_dates: List of training dates
            - val_data, val_dates, test_data, test_dates: Other splits
        ceemdan: Dictionary with CEEMDAN configuration:
            - trials: Number of noise trials (default: 100)
            - epsilon: Noise amplitude (default: 0.005)

    Returns:
        Dictionary containing:
            - decomposer: Fitted CEEMDANDecomposer object
            - train_imfs: np.ndarray of shape (n_imfs, n_train)
            - train_residual: np.ndarray of shape (n_train,)
            - n_imfs: Number of IMF components
    """
    train_data = split_data["train_data"]

    # Get CEEMDAN parameters
    trials = ceemdan.get("trials", 100)
    epsilon = ceemdan.get("epsilon", 0.005)

    logger.info("=" * 60)
    logger.info("CEEMDAN DECOMPOSITION - FIT ON TRAINING DATA")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train_data):,}")

    # Create and fit decomposer
    decomposer = CEEMDANDecomposer(trials=trials, epsilon=epsilon)
    train_imfs, train_residual = decomposer.fit_decompose(train_data)

    logger.info(f"Decomposition complete:")
    logger.info(f"  IMFs: {decomposer.n_imfs}")
    logger.info(f"  Train IMF shape: {train_imfs.shape}")
    logger.info(f"  Train residual shape: {train_residual.shape}")

    return {
        "decomposer": decomposer,
        "train_imfs": train_imfs,
        "train_residual": train_residual,
        "n_imfs": decomposer.n_imfs,
    }


def transform_val_test(
    split_data: Dict[str, Any],
    train_decomposition: Dict[str, Any]
) -> Dict[str, Any]:
    """Transform validation and test data using fitted decomposer.

    Uses the decomposer fitted on training data to decompose validation
    and test sets. IMF counts are aligned to match the training decomposition.

    Args:
        split_data: Dictionary from temporal_split
        train_decomposition: Dictionary from fit_decompose_train containing:
            - decomposer: Fitted CEEMDANDecomposer
            - train_imfs, train_residual: Training decomposition
            - n_imfs: Number of IMF components

    Returns:
        Dictionary containing:
            - train_imfs: np.ndarray of shape (n_imfs, n_train)
            - train_residual: np.ndarray of shape (n_train,)
            - train_dates: Training dates
            - val_imfs: np.ndarray of shape (n_imfs, n_val)
            - val_residual: np.ndarray of shape (n_val,)
            - val_dates: Validation dates
            - test_imfs: np.ndarray of shape (n_imfs, n_test)
            - test_residual: np.ndarray of shape (n_test,)
            - test_dates: Test dates
            - n_imfs: Number of IMF components
            - decomposer: The fitted decomposer object
    """
    decomposer = train_decomposition["decomposer"]
    val_data = split_data["val_data"]
    test_data = split_data["test_data"]

    logger.info("=" * 60)
    logger.info("CEEMDAN DECOMPOSITION - TRANSFORM VAL & TEST")
    logger.info("=" * 60)

    # Transform validation data
    logger.info(f"Transforming validation data (n={len(val_data):,})...")
    val_imfs, val_residual = decomposer.transform(val_data)

    # Transform test data
    logger.info(f"Transforming test data (n={len(test_data):,})...")
    test_imfs, test_residual = decomposer.transform(test_data)

    logger.info(f"Transform complete:")
    logger.info(f"  Val IMF shape: {val_imfs.shape}")
    logger.info(f"  Test IMF shape: {test_imfs.shape}")

    return {
        # Training (pass through from fit step)
        "train_imfs": train_decomposition["train_imfs"],
        "train_residual": train_decomposition["train_residual"],
        "train_dates": split_data["train_dates"],
        # Validation
        "val_imfs": val_imfs,
        "val_residual": val_residual,
        "val_dates": split_data["val_dates"],
        # Test
        "test_imfs": test_imfs,
        "test_residual": test_residual,
        "test_dates": split_data["test_dates"],
        "test_open_prices": split_data.get("test_open_prices"),  # For backtest
        # Metadata
        "n_imfs": train_decomposition["n_imfs"],
        "decomposer": decomposer,
    }
