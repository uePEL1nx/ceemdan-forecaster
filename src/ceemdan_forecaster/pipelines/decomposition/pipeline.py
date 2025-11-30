# -*- coding: utf-8 -*-
"""CEEMDAN decomposition pipeline definition.

This pipeline implements a fit/transform pattern for CEEMDAN decomposition:
1. Fit decomposer on training data only
2. Transform validation and test data using fitted decomposer

This ensures NO DATA LEAKAGE - the decomposer parameters (number of IMFs)
are determined solely from training data.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fit_decompose_train, transform_val_test


def create_pipeline(**kwargs) -> Pipeline:
    """Create the CEEMDAN decomposition pipeline.

    Pipeline structure:
    1. fit_decompose_train: Fit CEEMDAN on training data only
       - Inputs: split_data (from data_split), params:ceemdan
       - Outputs: train_decomposition

    2. transform_val_test: Transform val and test using fitted decomposer
       - Inputs: split_data, train_decomposition
       - Outputs: val_test_decomposition

    Returns:
        Pipeline with two nodes for fit/transform CEEMDAN
    """
    return pipeline([
        node(
            func=fit_decompose_train,
            inputs=["split_data", "params:ceemdan"],
            outputs="train_decomposition",
            name="fit_decompose_train",
            tags=["decomposition", "fit"],
        ),
        node(
            func=transform_val_test,
            inputs=["split_data", "train_decomposition"],
            outputs="val_test_decomposition",
            name="transform_val_test",
            tags=["decomposition", "transform"],
        ),
    ])
