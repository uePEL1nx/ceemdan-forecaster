# -*- coding: utf-8 -*-
"""IMF classification nodes.

This module implements Zero-Mean T-Test classification of IMFs into:
- H-IMF (High frequency): p-value >= alpha, cannot reject zero mean
  → These are random-like, modeled with Informer
- L-IMF (Low frequency): p-value < alpha, reject zero mean (has trend)
  → These are trend-like, modeled with LSTM

Reference: Li et al. (2024) Section 3, Module 1 (pages 13-14)
"""
import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def classify_imfs(
    train_decomposition: Dict[str, Any],
    ttest: Dict[str, Any]
) -> Dict[str, Any]:
    """Classify IMFs using Zero-Mean T-Test.

    Performs one-sample t-test on each IMF to determine if its mean
    is significantly different from zero.

    - H-IMF (p >= alpha): Cannot reject null hypothesis (mean = 0)
      These are high-frequency, random-like components → Informer model
    - L-IMF (p < alpha): Reject null hypothesis (mean != 0)
      These are low-frequency, trend-like components → LSTM model

    The residual is ALWAYS classified as L-IMF (per paper methodology).

    Args:
        train_decomposition: Dictionary from fit_decompose_train containing:
            - train_imfs: np.ndarray of shape (n_imfs, n_train)
            - train_residual: np.ndarray of shape (n_train,)
            - n_imfs: Number of IMF components
        ttest: Dictionary with t-test configuration:
            - alpha: Significance level (default: 0.10)

    Returns:
        Dictionary containing:
            - h_imf_indices: List of indices for H-IMF components (Informer)
            - l_imf_indices: List of indices for L-IMF components (LSTM)
            - imfi_index: Structural change point (first L-IMF)
            - test_results: List of dicts with p-values and classifications
            - n_imfs: Number of IMF components
            - alpha: Significance level used
    """
    train_imfs = train_decomposition["train_imfs"]
    train_residual = train_decomposition["train_residual"]
    n_imfs = train_decomposition["n_imfs"]

    alpha = ttest.get("alpha", 0.10)

    logger.info("=" * 60)
    logger.info("IMF CLASSIFICATION - Zero-Mean T-Test")
    logger.info("=" * 60)
    logger.info(f"Number of IMFs: {n_imfs}")
    logger.info(f"Significance level: alpha = {alpha}")

    test_results = []
    h_imf_indices = []
    l_imf_indices = []
    imfi_index = None  # Structural change point

    # Test each IMF
    for i in range(n_imfs):
        imf = train_imfs[i]

        # One-sample t-test: H0: mean = 0
        t_stat, p_value = stats.ttest_1samp(imf, 0)

        # Classification based on p-value
        reject_h0 = p_value < alpha
        classification = "L-IMF" if reject_h0 else "H-IMF"

        if reject_h0:
            l_imf_indices.append(i)
            # Track first L-IMF as structural change point
            if imfi_index is None:
                imfi_index = i
        else:
            h_imf_indices.append(i)

        # Significance stars for logging
        stars = ""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"

        logger.info(f"  IMF{i+1}: p={p_value:.6f}{stars:4s} -> {classification}")

        test_results.append({
            "component": f"IMF{i+1}",
            "index": i,
            "mean": float(np.mean(imf)),
            "std": float(np.std(imf)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_h0": reject_h0,
            "classification": classification,
        })

    # Test residual (always L-IMF per paper)
    t_stat, p_value = stats.ttest_1samp(train_residual, 0)

    logger.info(f"  RES:  p={p_value:.6f}     -> L-IMF (always)")

    test_results.append({
        "component": "RES",
        "index": n_imfs,  # Residual is after all IMFs
        "mean": float(np.mean(train_residual)),
        "std": float(np.std(train_residual)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "reject_h0": True,
        "classification": "L-IMF",
    })

    # Summary
    logger.info("-" * 60)
    if imfi_index is not None:
        logger.info(f"Structural change point: IMF{imfi_index + 1}")
        logger.info(f"  H-IMF (Informer): {len(h_imf_indices)} components - indices {h_imf_indices}")
        logger.info(f"  L-IMF (LSTM): {len(l_imf_indices)} components + RES - indices {l_imf_indices}")
    else:
        logger.warning("No structural change point found - all IMFs are H-IMF")
        logger.info(f"  H-IMF (Informer): {len(h_imf_indices)} components")
        logger.info(f"  L-IMF (LSTM): Residual only")

    return {
        "h_imf_indices": h_imf_indices,
        "l_imf_indices": l_imf_indices,
        "imfi_index": imfi_index,
        "test_results": test_results,
        "n_imfs": n_imfs,
        "alpha": alpha,
    }
