# -*- coding: utf-8 -*-
"""Gradio GUI for CEEMDAN-Informer-LSTM Pipeline.

This package provides a web-based interface for:
- Configuring pipeline parameters
- Running the Kedro pipeline
- Viewing MLflow results

Usage:
    python run_gui.py
    # Opens browser to http://localhost:7860
"""

from .app import create_app
from .config_manager import ConfigManager
from .presets import PRESETS, apply_preset
from .pipeline_runner import PipelineRunner
from .mlflow_client import MLflowClient

__all__ = [
    "create_app",
    "ConfigManager",
    "PRESETS",
    "apply_preset",
    "PipelineRunner",
    "MLflowClient",
]
