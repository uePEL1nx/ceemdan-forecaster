# -*- coding: utf-8 -*-
"""IMF classification pipeline definition.

This pipeline classifies IMFs into H-IMF (Informer) and L-IMF (LSTM)
using Zero-Mean T-Test.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import classify_imfs


def create_pipeline(**kwargs) -> Pipeline:
    """Create the IMF classification pipeline.

    Pipeline structure:
    - classify_imfs: Perform t-test on training IMFs
      - Inputs: train_decomposition (from decomposition), params:ttest
      - Outputs: classification

    Returns:
        Pipeline with classify_imfs node
    """
    return pipeline([
        node(
            func=classify_imfs,
            inputs=["train_decomposition", "params:ttest"],
            outputs="classification",
            name="classify_imfs",
            tags=["classification", "ttest"],
        ),
    ])
