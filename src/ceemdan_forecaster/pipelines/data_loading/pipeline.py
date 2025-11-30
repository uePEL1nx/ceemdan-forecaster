# -*- coding: utf-8 -*-
"""Data loading pipeline definition.

This pipeline loads time series data from CSV files for the
CEEMDAN-Informer-LSTM forecasting model.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_csv_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data loading pipeline.

    This pipeline has a single node that:
    1. Reads the data source configuration from parameters
    2. Loads the CSV file
    3. Extracts the time series data and dates
    4. Outputs to raw_data catalog entry

    Returns:
        Pipeline with load_csv_data node
    """
    return pipeline([
        node(
            func=load_csv_data,
            inputs="params:data_source",
            outputs="raw_data",
            name="load_csv_data",
            tags=["data", "ingestion"],
        ),
    ])
