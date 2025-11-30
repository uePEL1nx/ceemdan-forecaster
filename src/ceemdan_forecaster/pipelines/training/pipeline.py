# -*- coding: utf-8 -*-
"""Ensemble training pipeline definition.

This pipeline trains all 450 ensemble models:
- Informer models for H-IMF components (high-frequency)
- LSTM models for L-IMF components (low-frequency)
- Each component gets n_models (default: 50) models trained with different seeds

Training time depends on config preset:
- quick_test: ~15 min (5 models per component)
- standard: ~1 hour (20 models per component)
- production: ~2.5 hours (50 models per component)
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_all_ensembles


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ensemble training pipeline.

    Pipeline structure:
    - train_all_ensembles: Train all models for all components
      - Inputs: normalized_data, classification, params:informer, params:lstm,
                params:ensemble, params:runtime
      - Outputs: trained_models

    Returns:
        Pipeline with training node
    """
    return pipeline([
        node(
            func=train_all_ensembles,
            inputs=[
                "normalized_data",
                "classification",
                "params:informer",
                "params:lstm",
                "params:ensemble",
                "params:runtime",
            ],
            outputs="trained_models",
            name="train_all_ensembles",
            tags=["training", "gpu"],
        ),
    ])
