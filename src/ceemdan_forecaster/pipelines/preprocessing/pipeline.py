# -*- coding: utf-8 -*-
"""Preprocessing (normalization) pipeline definition.

This pipeline implements per-IMF zero-mean normalization:
1. Fit normalizers on training data only
2. Transform all splits using fitted normalizers

CRITICAL: Normalizers are fitted on training data ONLY to prevent data leakage.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fit_normalizers, transform_all_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the preprocessing (normalization) pipeline.

    Pipeline structure:
    1. fit_normalizers: Fit normalizers on training IMFs only
       - Inputs: val_test_decomposition (from decomposition)
       - Outputs: normalizers_data

    2. transform_all_data: Transform all splits using fitted normalizers
       - Inputs: val_test_decomposition, normalizers_data
       - Outputs: normalized_data

    Returns:
        Pipeline with two nodes for fit/transform normalization
    """
    return pipeline([
        node(
            func=fit_normalizers,
            inputs=["val_test_decomposition"],
            outputs="normalizers_data",
            name="fit_normalizers",
            tags=["preprocessing", "fit"],
        ),
        node(
            func=transform_all_data,
            inputs=["val_test_decomposition", "normalizers_data"],
            outputs="normalized_data",
            name="transform_all_data",
            tags=["preprocessing", "transform"],
        ),
    ])
