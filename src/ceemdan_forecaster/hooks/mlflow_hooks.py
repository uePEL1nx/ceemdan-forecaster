# -*- coding: utf-8 -*-
"""MLflow experiment tracking hooks."""
import logging
from typing import Any, Dict

import mlflow
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)


class MLflowTrackingHook:
    """Track experiments with MLflow.

    This hook automatically logs parameters, metrics, and artifacts
    to MLflow during pipeline execution.
    """

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Start an MLflow run before the pipeline begins.

        Args:
            run_params: Parameters passed to kedro run
            pipeline: The pipeline being executed
            catalog: The data catalog
        """
        mlflow.set_experiment("ceemdan_forecaster")
        mlflow.start_run(run_name=run_params.get("run_id"))
        logger.info(
            f"Started MLflow run: {mlflow.active_run().info.run_id}"
        )

    @hook_impl
    def after_node_run(
        self,
        node,
        catalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ):
        """Log node outputs to MLflow.

        Args:
            node: The node that just finished running
            catalog: The data catalog
            inputs: The inputs to the node
            outputs: The outputs from the node
        """
        # Log training metrics
        if "train" in node.name and isinstance(outputs, dict):
            if "history" in outputs:
                for key, val in outputs["history"].items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(f"{node.name}_{key}", val)

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        """End the MLflow run and log final metrics.

        Args:
            run_params: Parameters passed to kedro run
            pipeline: The pipeline that was executed
            catalog: The data catalog
        """
        # Try to log final evaluation metrics
        try:
            metrics = catalog.load("evaluation_metrics")
            if isinstance(metrics, dict):
                mlflow.log_metrics(metrics)
                logger.info(f"Logged final metrics: {metrics}")
        except Exception as e:
            logger.debug(f"Could not load evaluation_metrics: {e}")

        mlflow.end_run()
        logger.info("Ended MLflow run")

    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        """Handle pipeline errors and end the MLflow run.

        Args:
            error: The exception that was raised
            run_params: Parameters passed to kedro run
            pipeline: The pipeline that was executing
            catalog: The data catalog
        """
        mlflow.log_param("error", str(error)[:500])  # Truncate long errors
        mlflow.end_run(status="FAILED")
        logger.error(f"Pipeline failed: {error}")
