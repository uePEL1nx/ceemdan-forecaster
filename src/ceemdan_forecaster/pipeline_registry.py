# -*- coding: utf-8 -*-
"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from ceemdan_forecaster.pipelines import (
    data_loading,
    data_split,
    decomposition,
    classification,
    preprocessing,
    training,
    inference,
    evaluation,
    backtest,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines for the project.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Individual pipelines
    data_loading_pl = data_loading.create_pipeline()
    data_split_pl = data_split.create_pipeline()
    decomposition_pl = decomposition.create_pipeline()
    classification_pl = classification.create_pipeline()
    preprocessing_pl = preprocessing.create_pipeline()
    training_pl = training.create_pipeline()
    inference_pl = inference.create_pipeline()
    evaluation_pl = evaluation.create_pipeline()
    backtest_pl = backtest.create_pipeline()

    return {
        # Individual pipelines
        "data_loading": data_loading_pl,
        "data_split": data_split_pl,
        "decomposition": decomposition_pl,
        "classification": classification_pl,
        "preprocessing": preprocessing_pl,
        "training": training_pl,
        "inference": inference_pl,
        "evaluation": evaluation_pl,
        "backtest": backtest_pl,

        # Composite pipelines
        "data": data_loading_pl + data_split_pl,
        "prepare": (
            data_loading_pl + data_split_pl + decomposition_pl +
            classification_pl + preprocessing_pl
        ),
        "train_only": training_pl,
        "evaluate_only": inference_pl + evaluation_pl + backtest_pl,

        # Full pipeline
        "__default__": (
            data_loading_pl + data_split_pl + decomposition_pl +
            classification_pl + preprocessing_pl + training_pl +
            inference_pl + evaluation_pl + backtest_pl
        ),
    }
