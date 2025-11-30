# -*- coding: utf-8 -*-
"""Chart generation for Gradio GUI.

This module provides functions to load data and create interactive
Plotly charts for displaying pipeline results.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Any

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_equity_curve(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load equity curve CSV from data directory.

    Args:
        data_dir: Path to the data directory containing 08_reporting folder.

    Returns:
        DataFrame with equity curve data, or None if file doesn't exist.
    """
    csv_path = data_dir / "08_reporting" / "equity_curve.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def create_equity_chart(df: pd.DataFrame) -> Optional[Any]:
    """Create Plotly equity curve chart.

    Creates an interactive line chart comparing Strategy equity vs Buy & Hold.

    Args:
        df: DataFrame with 'strategy_equity' and 'buyhold_equity' columns.

    Returns:
        Plotly Figure object, or None if Plotly not available.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if df is None or df.empty:
        return None

    fig = go.Figure()

    # Strategy equity line (blue)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['strategy_equity'],
        name='Strategy',
        line=dict(color='#2196F3', width=2),
        hovertemplate='Strategy: $%{y:,.0f}<extra></extra>'
    ))

    # Buy & Hold equity line (gray)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['buyhold_equity'],
        name='Buy & Hold',
        line=dict(color='#9E9E9E', width=2),
        hovertemplate='Buy & Hold: $%{y:,.0f}<extra></extra>'
    ))

    # Add shaded region for outperformance
    # Green where strategy > buyhold, red where strategy < buyhold
    strategy = df['strategy_equity'].values
    buyhold = df['buyhold_equity'].values

    # Fill between where strategy outperforms
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(strategy) + list(buyhold[::-1]),
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.1)',  # Light green
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='Strategy vs Buy & Hold Equity Curve',
            font=dict(size=16)
        ),
        xaxis_title='Trading Day',
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
