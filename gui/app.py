# -*- coding: utf-8 -*-
"""Gradio GUI for CEEMDAN-Informer-LSTM Pipeline.

This module provides the main Gradio application interface for:
- Configuring pipeline parameters
- Applying presets
- Running the Kedro pipeline
- Viewing MLflow results and charts
"""
import gradio as gr
from pathlib import Path

from .config_manager import ConfigManager
from .presets import PRESETS, apply_preset, PRESET_DESCRIPTIONS
from .pipeline_runner import PipelineRunner
from .mlflow_client import MLflowClient
from .charts import load_equity_curve, create_equity_chart, create_empty_chart


# Initialize components (will be created when app starts)
config_mgr = None
runner = None
mlflow_client = None


def init_components():
    """Initialize global components."""
    global config_mgr, runner, mlflow_client
    config_mgr = ConfigManager()
    runner = PipelineRunner()
    mlflow_client = MLflowClient()


def load_current_config():
    """Load and return current configuration."""
    if config_mgr is None:
        init_components()
    return config_mgr.load()


def apply_preset_handler(preset_name: str):
    """Apply a preset and return updated values."""
    if config_mgr is None:
        init_components()
    params = config_mgr.load()
    params = apply_preset(params, preset_name)
    config_mgr.save(params)
    return extract_ui_values(params) + (f"Applied preset: {preset_name}",)


def extract_ui_values(params: dict):
    """Extract values for UI components from params dict."""
    return (
        # Main params
        params.get("data_source", {}).get("path", ""),
        params.get("data_split", {}).get("train_ratio", 0.80),
        params.get("data_split", {}).get("val_ratio", 0.10),
        params.get("data_split", {}).get("test_ratio", 0.10),
        params.get("ensemble", {}).get("n_models", 50),
        params.get("runtime", {}).get("device", "auto"),
        # CEEMDAN
        params.get("ceemdan", {}).get("trials", 100),
        params.get("ceemdan", {}).get("epsilon", 0.005),
        params.get("ceemdan", {}).get("max_imf", -1),
        # Informer
        params.get("informer", {}).get("seq_len", 96),
        params.get("informer", {}).get("label_len", 48),
        params.get("informer", {}).get("d_model", 256),
        params.get("informer", {}).get("n_heads", 8),
        params.get("informer", {}).get("e_layers", 2),
        params.get("informer", {}).get("dropout", 0.05),
        params.get("informer", {}).get("epochs", 10),
        params.get("informer", {}).get("learning_rate", 0.0001),
        params.get("informer", {}).get("patience", 3),
        # LSTM
        params.get("lstm", {}).get("look_back", 20),
        params.get("lstm", {}).get("hidden_size", 4),
        params.get("lstm", {}).get("num_layers", 1),
        params.get("lstm", {}).get("dropout", 0.1),
        params.get("lstm", {}).get("epochs", 100),
        params.get("lstm", {}).get("learning_rate", 0.001),
        params.get("lstm", {}).get("patience", 10),
        # Backtest
        params.get("backtest", {}).get("transaction_cost", 0.001),
        params.get("backtest", {}).get("initial_capital", 100000),
        params.get("backtest", {}).get("signal_threshold", 0.0),
        params.get("backtest", {}).get("execution_timing", "close"),
    )


def save_config(
    data_path, train_ratio, val_ratio, test_ratio, n_models, device,
    ceemdan_trials, ceemdan_epsilon, ceemdan_max_imf,
    inf_seq_len, inf_label_len, inf_d_model, inf_n_heads, inf_e_layers,
    inf_dropout, inf_epochs, inf_lr, inf_patience,
    lstm_look_back, lstm_hidden, lstm_layers, lstm_dropout,
    lstm_epochs, lstm_lr, lstm_patience,
    bt_cost, bt_capital, bt_threshold, bt_timing
):
    """Save all UI values to parameters.yml."""
    if config_mgr is None:
        init_components()

    params = config_mgr.load()

    # Update data_source
    if "data_source" not in params:
        params["data_source"] = {}
    params["data_source"]["path"] = data_path

    # Update data_split
    if "data_split" not in params:
        params["data_split"] = {}
    params["data_split"]["train_ratio"] = float(train_ratio)
    params["data_split"]["val_ratio"] = float(val_ratio)
    params["data_split"]["test_ratio"] = float(test_ratio)

    # Update ensemble
    if "ensemble" not in params:
        params["ensemble"] = {}
    params["ensemble"]["n_models"] = int(n_models)

    # Update runtime
    if "runtime" not in params:
        params["runtime"] = {}
    params["runtime"]["device"] = device

    # Update CEEMDAN
    if "ceemdan" not in params:
        params["ceemdan"] = {}
    params["ceemdan"]["trials"] = int(ceemdan_trials)
    params["ceemdan"]["epsilon"] = float(ceemdan_epsilon)
    params["ceemdan"]["max_imf"] = int(ceemdan_max_imf)

    # Update Informer
    if "informer" not in params:
        params["informer"] = {}
    params["informer"]["seq_len"] = int(inf_seq_len)
    params["informer"]["label_len"] = int(inf_label_len)
    params["informer"]["d_model"] = int(inf_d_model)
    params["informer"]["n_heads"] = int(inf_n_heads)
    params["informer"]["e_layers"] = int(inf_e_layers)
    params["informer"]["dropout"] = float(inf_dropout)
    params["informer"]["epochs"] = int(inf_epochs)
    params["informer"]["learning_rate"] = float(inf_lr)
    params["informer"]["patience"] = int(inf_patience)

    # Update LSTM
    if "lstm" not in params:
        params["lstm"] = {}
    params["lstm"]["look_back"] = int(lstm_look_back)
    params["lstm"]["hidden_size"] = int(lstm_hidden)
    params["lstm"]["num_layers"] = int(lstm_layers)
    params["lstm"]["dropout"] = float(lstm_dropout)
    params["lstm"]["epochs"] = int(lstm_epochs)
    params["lstm"]["learning_rate"] = float(lstm_lr)
    params["lstm"]["patience"] = int(lstm_patience)

    # Update backtest
    if "backtest" not in params:
        params["backtest"] = {}
    params["backtest"]["transaction_cost"] = float(bt_cost)
    params["backtest"]["initial_capital"] = int(bt_capital)
    params["backtest"]["signal_threshold"] = float(bt_threshold)
    params["backtest"]["execution_timing"] = bt_timing

    config_mgr.save(params)
    return "Configuration saved to conf/base/parameters.yml"


def refresh_equity_chart():
    """Load and create equity curve chart."""
    data_dir = Path(__file__).parent.parent / "data"
    df = load_equity_curve(data_dir)
    if df is None:
        return create_empty_chart()
    return create_equity_chart(df)


def run_pipeline_handler():
    """Run the pipeline and stream output, then refresh charts."""
    if runner is None:
        init_components()

    output = ""
    for line in runner.run():
        output += line + "\n"
        # During pipeline execution, just update log
        yield output

    # After completion, the log is finalized
    yield output


def cancel_pipeline_handler():
    """Cancel running pipeline."""
    if runner is None:
        init_components()
    return runner.cancel()


def get_results_handler():
    """Fetch latest results from MLflow."""
    if mlflow_client is None:
        init_components()
    result = mlflow_client.get_latest_run()
    return mlflow_client.format_metrics(result)


def refresh_all_results():
    """Refresh both metrics and chart."""
    metrics = get_results_handler()
    chart = refresh_equity_chart()
    return metrics, chart


def create_app():
    """Create the Gradio application."""
    init_components()

    with gr.Blocks(title="CEEMDAN Pipeline") as app:
        gr.Markdown("# CEEMDAN-Informer-LSTM Pipeline")
        gr.Markdown("Configure parameters, run the pipeline, and view results.")

        with gr.Tabs():
            # ===== CONFIGURATION TAB =====
            with gr.TabItem("Configuration"):
                # Status bar
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    max_lines=1,
                )

                # Preset buttons
                gr.Markdown("### Presets")
                with gr.Row():
                    btn_quick = gr.Button(
                        "Quick Test (~15 min)",
                        variant="secondary",
                        elem_classes=["preset-btn"]
                    )
                    btn_standard = gr.Button(
                        "Standard (~1 hour)",
                        variant="secondary",
                        elem_classes=["preset-btn"]
                    )
                    btn_production = gr.Button(
                        "Production (~2.5 hours)",
                        variant="primary",
                        elem_classes=["preset-btn"]
                    )
                    btn_load = gr.Button(
                        "Load Current",
                        variant="secondary",
                        elem_classes=["preset-btn"]
                    )

                # Main parameters (always visible)
                gr.Markdown("### Main Parameters")
                with gr.Group():
                    with gr.Row():
                        data_path = gr.Textbox(
                            label="Data Source Path",
                            scale=3,
                            placeholder="C:/Users/jd/LTSM/data/raw/file.csv"
                        )
                    with gr.Row():
                        train_ratio = gr.Number(label="Train Ratio", value=0.80, precision=2, minimum=0.5, maximum=0.95)
                        val_ratio = gr.Number(label="Val Ratio", value=0.10, precision=2, minimum=0.05, maximum=0.25)
                        test_ratio = gr.Number(label="Test Ratio", value=0.10, precision=2, minimum=0.05, maximum=0.25)
                    with gr.Row():
                        n_models = gr.Number(label="Ensemble Models", value=50, precision=0, minimum=1, maximum=100)
                        device = gr.Dropdown(
                            label="Device",
                            choices=["auto", "cuda", "cpu"],
                            value="auto"
                        )

                # CEEMDAN Accordion
                with gr.Accordion("CEEMDAN Parameters", open=False):
                    with gr.Row():
                        ceemdan_trials = gr.Number(label="Trials", value=100, precision=0, minimum=10, maximum=500)
                        ceemdan_epsilon = gr.Number(label="Epsilon", value=0.005, precision=4, minimum=0.001, maximum=0.1)
                        ceemdan_max_imf = gr.Number(label="Max IMF (-1=auto)", value=-1, precision=0, minimum=-1, maximum=20)

                # Informer Accordion
                with gr.Accordion("Informer Parameters (H-IMF)", open=False):
                    with gr.Row():
                        inf_seq_len = gr.Number(label="Sequence Length", value=96, precision=0, minimum=24, maximum=512)
                        inf_label_len = gr.Number(label="Label Length", value=48, precision=0, minimum=12, maximum=256)
                        inf_d_model = gr.Number(label="d_model", value=256, precision=0, minimum=32, maximum=1024)
                        inf_n_heads = gr.Number(label="Attention Heads", value=8, precision=0, minimum=1, maximum=16)
                    with gr.Row():
                        inf_e_layers = gr.Number(label="Encoder Layers", value=2, precision=0, minimum=1, maximum=6)
                        inf_dropout = gr.Number(label="Dropout", value=0.05, precision=2, minimum=0.0, maximum=0.5)
                        inf_epochs = gr.Number(label="Epochs", value=10, precision=0, minimum=1, maximum=100)
                    with gr.Row():
                        inf_lr = gr.Number(label="Learning Rate", value=0.0001, precision=6, minimum=0.000001, maximum=0.01)
                        inf_patience = gr.Number(label="Early Stop Patience", value=3, precision=0, minimum=1, maximum=20)

                # LSTM Accordion
                with gr.Accordion("LSTM Parameters (L-IMF)", open=False):
                    with gr.Row():
                        lstm_look_back = gr.Number(label="Look Back", value=20, precision=0, minimum=5, maximum=100)
                        lstm_hidden = gr.Number(label="Hidden Size", value=4, precision=0, minimum=1, maximum=256)
                        lstm_layers = gr.Number(label="Num Layers", value=1, precision=0, minimum=1, maximum=4)
                    with gr.Row():
                        lstm_dropout = gr.Number(label="Dropout", value=0.1, precision=2, minimum=0.0, maximum=0.5)
                        lstm_epochs = gr.Number(label="Epochs", value=100, precision=0, minimum=1, maximum=500)
                    with gr.Row():
                        lstm_lr = gr.Number(label="Learning Rate", value=0.001, precision=6, minimum=0.000001, maximum=0.1)
                        lstm_patience = gr.Number(label="Early Stop Patience", value=10, precision=0, minimum=1, maximum=50)

                # Backtest Accordion
                with gr.Accordion("Backtest Parameters", open=False):
                    with gr.Row():
                        bt_cost = gr.Number(label="Transaction Cost", value=0.001, precision=4, minimum=0.0, maximum=0.01)
                        bt_capital = gr.Number(label="Initial Capital", value=100000, precision=0, minimum=1000, maximum=10000000)
                    with gr.Row():
                        bt_threshold = gr.Number(label="Signal Threshold", value=0.0, precision=4, minimum=-100, maximum=100)
                        bt_timing = gr.Dropdown(
                            label="Execution Timing",
                            choices=["close", "next_open"],
                            value="close"
                        )

                # Action buttons
                gr.Markdown("---")
                with gr.Row():
                    btn_run = gr.Button(
                        "Run Pipeline",
                        variant="primary",
                        scale=2,
                        elem_classes=["run-btn"]
                    )
                    btn_cancel = gr.Button("Cancel", variant="stop")
                    btn_save = gr.Button("Save Config", variant="secondary")

                # Execution log
                with gr.Accordion("Execution Log", open=True):
                    log_output = gr.Textbox(
                        label="Pipeline Output",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                    )

            # ===== RESULTS TAB =====
            with gr.TabItem("Results"):
                with gr.Row():
                    btn_refresh = gr.Button("Refresh Results", variant="primary")
                    gr.Markdown("*Results auto-refresh after pipeline completion*")

                with gr.Tabs():
                    with gr.TabItem("Metrics"):
                        results_text = gr.Textbox(
                            label="Latest Run Metrics (from MLflow)",
                            lines=20,
                            interactive=False,
                        )

                    with gr.TabItem("Equity Curve"):
                        equity_chart = gr.Plot(
                            label="Strategy vs Buy & Hold",
                            show_label=True
                        )

        # Collect all input components for save/load
        all_inputs = [
            data_path, train_ratio, val_ratio, test_ratio, n_models, device,
            ceemdan_trials, ceemdan_epsilon, ceemdan_max_imf,
            inf_seq_len, inf_label_len, inf_d_model, inf_n_heads, inf_e_layers,
            inf_dropout, inf_epochs, inf_lr, inf_patience,
            lstm_look_back, lstm_hidden, lstm_layers, lstm_dropout,
            lstm_epochs, lstm_lr, lstm_patience,
            bt_cost, bt_capital, bt_threshold, bt_timing,
        ]

        # All outputs including status
        all_outputs_with_status = all_inputs + [status_text]

        # Wire up preset buttons
        btn_quick.click(
            fn=lambda: apply_preset_handler("quick_test"),
            outputs=all_outputs_with_status
        )
        btn_standard.click(
            fn=lambda: apply_preset_handler("standard"),
            outputs=all_outputs_with_status
        )
        btn_production.click(
            fn=lambda: apply_preset_handler("production"),
            outputs=all_outputs_with_status
        )
        btn_load.click(
            fn=lambda: extract_ui_values(load_current_config()) + ("Loaded current configuration",),
            outputs=all_outputs_with_status
        )

        # Wire up action buttons
        btn_save.click(fn=save_config, inputs=all_inputs, outputs=status_text)
        btn_run.click(fn=run_pipeline_handler, outputs=log_output)
        btn_cancel.click(fn=cancel_pipeline_handler, outputs=status_text)

        # Wire up results refresh
        btn_refresh.click(
            fn=refresh_all_results,
            outputs=[results_text, equity_chart]
        )

        # Load current config and initial chart on startup
        app.load(
            fn=lambda: extract_ui_values(load_current_config()) + ("Ready",),
            outputs=all_outputs_with_status
        )
        app.load(
            fn=refresh_equity_chart,
            outputs=equity_chart
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860, share=False)
