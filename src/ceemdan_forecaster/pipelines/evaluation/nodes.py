# -*- coding: utf-8 -*-
"""Evaluation metrics nodes.

This module calculates performance metrics for the ensemble predictions:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

Targets from paper:
- MAE < 15
- RMSE < 20
- R² > 0.99
"""
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def calculate_metrics(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate evaluation metrics for ensemble predictions.

    This function computes standard regression metrics comparing
    the ensemble predictions against actual values.

    Args:
        predictions: Dictionary containing:
            - predictions: numpy array of predicted values
            - actual: numpy array of actual values
            - test_dates: list of dates for test samples
            - n_samples: number of samples

    Returns:
        Dictionary containing:
            - mae: Mean Absolute Error
            - rmse: Root Mean Square Error
            - mape: Mean Absolute Percentage Error (%)
            - r2: R-squared (Coefficient of Determination)
            - n_samples: Number of samples evaluated
            - prediction_stats: Summary statistics for predictions
            - actual_stats: Summary statistics for actual values
            - targets: Target values from paper for comparison
            - targets_met: Boolean flags for each target
    """
    logger.info("Calculating evaluation metrics...")

    # Extract predictions and actual values
    pred = np.array(predictions["predictions"])
    actual = np.array(predictions["actual"])
    n_samples = len(pred)

    logger.info(f"Evaluating {n_samples} samples")
    logger.info(f"Prediction range: {pred.min():.2f} to {pred.max():.2f}")
    logger.info(f"Actual range: {actual.min():.2f} to {actual.max():.2f}")

    # Calculate MAE (Mean Absolute Error)
    mae = float(np.mean(np.abs(pred - actual)))

    # Calculate RMSE (Root Mean Square Error)
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = actual != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
    else:
        mape = float("inf")
        logger.warning("Cannot calculate MAPE: all actual values are zero")

    # Calculate R² (Coefficient of Determination)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot > 0:
        r2 = float(1 - (ss_res / ss_tot))
    else:
        r2 = float("nan")
        logger.warning("Cannot calculate R²: variance of actual values is zero")

    # Log results
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"MAE:  {mae:.4f} (target: < 15)")
    logger.info(f"RMSE: {rmse:.4f} (target: < 20)")
    logger.info(f"MAPE: {mape:.2f}% (target: < 0.5%)")
    logger.info(f"R²:   {r2:.4f} (target: > 0.99)")
    logger.info("=" * 60)

    # Check targets
    mae_target = 15.0
    rmse_target = 20.0
    r2_target = 0.99

    mae_met = mae < mae_target
    rmse_met = rmse < rmse_target
    r2_met = r2 > r2_target

    logger.info("TARGET STATUS:")
    logger.info(f"  MAE < {mae_target}: {'PASS' if mae_met else 'FAIL'}")
    logger.info(f"  RMSE < {rmse_target}: {'PASS' if rmse_met else 'FAIL'}")
    logger.info(f"  R² > {r2_target}: {'PASS' if r2_met else 'FAIL'}")

    # Calculate summary statistics
    prediction_stats = {
        "min": float(pred.min()),
        "max": float(pred.max()),
        "mean": float(pred.mean()),
        "std": float(pred.std()),
    }

    actual_stats = {
        "min": float(actual.min()),
        "max": float(actual.max()),
        "mean": float(actual.mean()),
        "std": float(actual.std()),
    }

    # Error statistics
    errors = pred - actual
    error_stats = {
        "min": float(errors.min()),
        "max": float(errors.max()),
        "mean": float(errors.mean()),
        "std": float(errors.std()),
        "median": float(np.median(errors)),
    }

    # Build result dictionary (JSON-serializable)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "n_samples": n_samples,
        "prediction_stats": prediction_stats,
        "actual_stats": actual_stats,
        "error_stats": error_stats,
        "targets": {
            "mae": mae_target,
            "rmse": rmse_target,
            "r2": r2_target,
        },
        "targets_met": {
            "mae": mae_met,
            "rmse": rmse_met,
            "r2": r2_met,
            "all": mae_met and rmse_met and r2_met,
        },
    }

    # Log final summary
    all_met = metrics["targets_met"]["all"]
    if all_met:
        logger.info("ALL TARGETS MET!")
    else:
        logger.info("Some targets not met - see metrics for details")

    return metrics
