# -*- coding: utf-8 -*-
"""MLflow client for fetching results.

Provides functionality to fetch experiment results from MLflow
for display in the GUI.
"""
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


class MLflowClient:
    """Fetches results from MLflow tracking."""

    def __init__(self, tracking_uri: str = None):
        """Initialize MLflowClient.

        Args:
            tracking_uri: MLflow tracking URI. If None, uses default mlruns directory.
        """
        if tracking_uri is None:
            # Use file:// URI format for Windows compatibility
            mlruns_path = Path(__file__).parent.parent / "mlruns"
            tracking_uri = mlruns_path.as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = "ceemdan_forecaster"

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the latest run.

        Returns:
            Dictionary with run info and metrics, or None if no runs found.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                return None

            run = runs.iloc[0]

            # Extract metrics
            metrics = {}
            for col in runs.columns:
                if col.startswith("metrics."):
                    key = col.replace("metrics.", "")
                    value = run[col]
                    # Filter out NaN values
                    if value is not None and not (isinstance(value, float) and pd.isna(value)):
                        metrics[key] = value

            return {
                "run_id": run["run_id"],
                "status": run["status"],
                "start_time": run["start_time"],
                "end_time": run.get("end_time"),
                "metrics": metrics,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recent_runs(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get recent runs.

        Args:
            max_results: Maximum number of runs to return.

        Returns:
            List of run dictionaries.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=max_results,
            )

            if runs.empty:
                return []

            result = []
            for _, run in runs.iterrows():
                metrics = {}
                for col in runs.columns:
                    if col.startswith("metrics."):
                        key = col.replace("metrics.", "")
                        value = run[col]
                        if value is not None and not (isinstance(value, float) and pd.isna(value)):
                            metrics[key] = value

                result.append({
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "start_time": run["start_time"],
                    "metrics": metrics,
                })

            return result
        except Exception:
            return []

    def format_metrics(self, result: Dict[str, Any]) -> str:
        """Format metrics for display.

        Args:
            result: Run result dictionary from get_latest_run().

        Returns:
            Formatted string for display.
        """
        if result is None:
            return "No runs found in MLflow"

        if "error" in result:
            return f"Error fetching MLflow data: {result['error']}"

        metrics = result.get("metrics", {})
        if not metrics:
            return (
                f"Run ID: {result.get('run_id', 'Unknown')[:8]}...\n"
                f"Status: {result.get('status', 'Unknown')}\n\n"
                "No metrics logged for this run.\n"
                "Metrics are logged when the evaluation pipeline runs.\n"
                "Run the full pipeline or 'evaluate_only' to see metrics."
            )

        lines = []
        lines.append(f"Run ID: {result.get('run_id', 'Unknown')[:8]}...")
        lines.append(f"Status: {result.get('status', 'Unknown')}")
        lines.append("")
        lines.append("Metrics:")
        lines.append("-" * 30)

        # Group metrics by category
        eval_metrics = {}
        backtest_metrics = {}
        other_metrics = {}

        for key, value in metrics.items():
            if key in ['MAE', 'RMSE', 'MAPE', 'R2']:
                eval_metrics[key] = value
            elif key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                backtest_metrics[key] = value
            else:
                other_metrics[key] = value

        # Format evaluation metrics
        if eval_metrics:
            lines.append("Evaluation:")
            for key, value in eval_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        # Format backtest metrics
        if backtest_metrics:
            lines.append("\nBacktest:")
            for key, value in backtest_metrics.items():
                if isinstance(value, float):
                    if 'return' in key.lower() or 'rate' in key.lower():
                        lines.append(f"  {key}: {value*100:.2f}%")
                    else:
                        lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        # Format other metrics
        if other_metrics:
            lines.append("\nOther:")
            for key, value in other_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)
