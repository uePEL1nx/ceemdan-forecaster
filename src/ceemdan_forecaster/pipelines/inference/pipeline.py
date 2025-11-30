# -*- coding: utf-8 -*-
"""Ensemble inference pipeline definition.

This pipeline generates predictions using trained ensemble models:
1. Load trained models and generate component predictions
2. Average predictions across seeds (ensemble)
3. Denormalize and reconstruct final forecast
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ensemble_predict, reconstruct_predictions


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ensemble inference pipeline.

    Pipeline structure:
    1. ensemble_predict: Generate predictions for all components
       - Inputs: trained_models, normalized_data, params:informer, params:lstm, params:runtime
       - Outputs: component_predictions

    2. reconstruct_predictions: Denormalize and sum components
       - Inputs: component_predictions, normalized_data, val_test_decomposition
       - Outputs: predictions

    Returns:
        Pipeline with inference nodes
    """
    return pipeline([
        node(
            func=ensemble_predict,
            inputs=[
                "trained_models",
                "normalized_data",
                "params:informer",
                "params:lstm",
                "params:runtime",
            ],
            outputs="component_predictions",
            name="ensemble_predict",
            tags=["inference", "gpu"],
        ),
        node(
            func=reconstruct_predictions,
            inputs=[
                "component_predictions",
                "normalized_data",
                "val_test_decomposition",
            ],
            outputs="predictions",
            name="reconstruct_predictions",
            tags=["inference", "reconstruction"],
        ),
    ])
