# -*- coding: utf-8 -*-
"""Backtesting pipeline definition.

This pipeline runs backtesting simulation on ensemble predictions:
1. Generate trading signals from predictions
2. Run backtest simulation with equity curve
3. Calculate performance metrics (Sharpe, Max DD, etc.)
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_signals, run_backtest


def create_pipeline(**kwargs) -> Pipeline:
    """Create the backtesting pipeline.

    Pipeline structure:
    1. generate_signals: Create trading signals from predictions
       - Inputs: predictions, params:backtest
       - Outputs: signal_data (intermediate)

    2. run_backtest: Run simulation and calculate metrics
       - Inputs: signal_data
       - Outputs: backtest_results, backtest_metrics, equity_curve

    Returns:
        Pipeline with backtesting nodes
    """
    return pipeline([
        node(
            func=generate_signals,
            inputs=["predictions", "params:backtest"],
            outputs="signal_data",
            name="generate_signals",
            tags=["backtest", "signals"],
        ),
        node(
            func=run_backtest,
            inputs="signal_data",
            outputs=["backtest_results", "backtest_metrics", "equity_curve"],
            name="run_backtest",
            tags=["backtest", "simulation"],
        ),
    ])
