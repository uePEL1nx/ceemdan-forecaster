# -*- coding: utf-8 -*-
"""Data split pipeline definition.

This pipeline splits the raw time series data into train/val/test sets
using strict temporal ordering to prevent data leakage.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import temporal_split


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data split pipeline.

    This pipeline has a single node that:
    1. Reads raw_data from the previous pipeline
    2. Reads split configuration from parameters
    3. Performs temporal split (80/10/10)
    4. Outputs split_data catalog entry

    Returns:
        Pipeline with temporal_split node
    """
    return pipeline([
        node(
            func=temporal_split,
            inputs=["raw_data", "params:data_split"],
            outputs="split_data",
            name="temporal_split",
            tags=["data", "split"],
        ),
    ])
