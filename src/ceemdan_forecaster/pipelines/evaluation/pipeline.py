# -*- coding: utf-8 -*-
"""Evaluation metrics pipeline definition.

This pipeline calculates performance metrics for ensemble predictions:
1. Calculate MAE, RMSE, MAPE, R² against actual values
2. Compare against paper targets (MAE<15, RMSE<20, R²>0.99)
3. Output metrics as JSON for reporting
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    """Create the evaluation pipeline.

    Pipeline structure:
    1. calculate_metrics: Compute regression metrics
       - Inputs: predictions (from inference pipeline)
       - Outputs: evaluation_metrics (JSON)

    Returns:
        Pipeline with evaluation node
    """
    return pipeline([
        node(
            func=calculate_metrics,
            inputs="predictions",
            outputs="evaluation_metrics",
            name="calculate_metrics",
            tags=["evaluation", "metrics"],
        ),
    ])
