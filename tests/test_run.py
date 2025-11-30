# -*- coding: utf-8 -*-
"""
This module contains tests for the ceemdan_forecaster Kedro project.
"""
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import pipelines


class TestKedroProject:
    """Test basic project setup and pipeline registration."""

    def test_project_bootstrap(self):
        """Test that the project can be bootstrapped."""
        bootstrap_project(Path.cwd())
        # If we get here without exception, bootstrap succeeded

    def test_pipelines_registered(self):
        """Test that all pipelines are registered."""
        bootstrap_project(Path.cwd())

        # Check all expected pipelines exist
        expected = [
            "__default__",
            "data_loading",
            "data_split",
            "decomposition",
            "classification",
            "preprocessing",
            "training",
            "inference",
            "evaluation",
            "backtest",
            "data",
            "prepare",
            "train_only",
            "evaluate_only",
        ]

        for pipeline_name in expected:
            assert pipeline_name in pipelines, f"Pipeline '{pipeline_name}' not found"

    def test_data_loading_pipeline_has_nodes(self):
        """Test that the data_loading pipeline has at least one node."""
        bootstrap_project(Path.cwd())

        pipeline = pipelines.get("data_loading")
        assert pipeline is not None
        # data_loading should have the load_csv_data node
        assert len(pipeline.nodes) >= 1
