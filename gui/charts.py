# -*- coding: utf-8 -*-
"""Chart generation for Gradio GUI.

This module provides functions to load data and create interactive
Plotly charts for displaying pipeline results.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Any, Tuple

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def get_instrument_name(config_dir: Path) -> str:
    """Extract instrument name from parameters.yml.

    Args:
        config_dir: Path to the conf directory.

    Returns:
        Instrument name extracted from data source path, or 'Unknown' if not found.
    """
    params_path = config_dir / "base" / "parameters.yml"
    if not params_path.exists():
        return "Unknown"

    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        data_path = params.get("data_source", {}).get("path", "")
        if data_path:
            # Extract filename without extension
            filename = Path(data_path).stem
            # Clean up the name (remove $ prefix if present)
            return filename.replace("$", "").replace("_", " ")
        return "Unknown"
    except Exception:
        return "Unknown"


def load_equity_curve(data_dir: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Load equity curve CSV from data directory.

    Args:
        data_dir: Path to the data directory containing 08_reporting folder.

    Returns:
        Tuple of (DataFrame with equity curve data, instrument name).
        DataFrame is None if file doesn't exist.
    """
    csv_path = data_dir / "08_reporting" / "equity_curve.csv"

    # Get instrument name from config
    config_dir = data_dir.parent / "conf"
    instrument_name = get_instrument_name(config_dir)

    if not csv_path.exists():
        return None, instrument_name
    try:
        df = pd.read_csv(csv_path)
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            # Try to parse dates - they might be indices or actual dates
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                pass  # Keep as-is if conversion fails
        return df, instrument_name
    except Exception:
        return None, instrument_name


def create_equity_chart(df: pd.DataFrame, instrument_name: str = "Unknown") -> Optional[Any]:
    """Create Plotly equity curve chart.

    Creates an interactive line chart comparing Strategy equity vs Buy & Hold.

    Args:
        df: DataFrame with 'strategy_equity', 'buyhold_equity', and optionally 'date' columns.
        instrument_name: Name of the instrument being traded.

    Returns:
        Plotly Figure object, or None if Plotly not available.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if df is None or df.empty:
        return None

    fig = go.Figure()

    # Determine x-axis: use 'date' column if available, otherwise use index
    if 'date' in df.columns:
        x_values = df['date']
        x_axis_title = 'Date'
        # Format for hover based on whether dates are datetime or not
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            date_hover = '%{x|%Y-%m-%d}'
        else:
            date_hover = '%{x}'
    else:
        x_values = df.index
        x_axis_title = 'Trading Day'
        date_hover = 'Day %{x}'

    # Strategy equity line (blue)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df['strategy_equity'],
        name='Strategy',
        line=dict(color='#2196F3', width=2),
        hovertemplate=f'{date_hover}<br>Strategy: $%{{y:,.0f}}<extra></extra>'
    ))

    # Buy & Hold equity line (gray)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df['buyhold_equity'],
        name='Buy & Hold',
        line=dict(color='#9E9E9E', width=2),
        hovertemplate=f'{date_hover}<br>Buy & Hold: $%{{y:,.0f}}<extra></extra>'
    ))

    # Add shaded region for outperformance
    # Green where strategy > buyhold, red where strategy < buyhold
    strategy = df['strategy_equity'].values
    buyhold = df['buyhold_equity'].values

    # Fill between where strategy outperforms
    fig.add_trace(go.Scatter(
        x=list(x_values) + list(x_values[::-1]),
        y=list(strategy) + list(buyhold[::-1]),
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.1)',  # Light green
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Layout with instrument name in title
    fig.update_layout(
        title=dict(
            text=f'{instrument_name} - Strategy vs Buy & Hold',
            font=dict(size=16)
        ),
        xaxis_title=x_axis_title,
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        template='plotly_white',
        height=450,
        margin=dict(l=60, r=30, t=50, b=50),
        yaxis=dict(
            tickformat='$,.0f',
            gridcolor='rgba(128,128,128,0.2)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)'
        )
    )

    return fig


def create_empty_chart() -> Optional[Any]:
    """Create an empty placeholder chart.

    Returns:
        Plotly Figure with message indicating no data available.
    """
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()
    fig.add_annotation(
        text="No data available. Run the pipeline first.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color='gray')
    )
    fig.update_layout(
        height=450,
        template='plotly_white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig
