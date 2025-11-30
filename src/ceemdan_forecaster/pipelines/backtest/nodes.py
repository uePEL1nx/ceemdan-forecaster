# -*- coding: utf-8 -*-
"""Backtesting nodes.

This module implements a simple long-only trading strategy based on predictions:
- Long (signal=1) when predicted direction is UP (pred[i] > pred[i-1])
- Flat (signal=0) otherwise

The signal uses PREDICTED DIRECTION rather than absolute price comparison
because models may have systematic bias in absolute price levels while
still accurately predicting direction (up/down movements).

Execution timing options:
- "close": Enter at current day's close price (same-day execution)
- "next_open": Enter at next day's price (simulates overnight/next-day execution)

Performance metrics calculated:
- Total Return
- Annualized Return
- Sharpe Ratio
- Volatility
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor
"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252


def generate_signals(
    predictions: Dict[str, Any],
    backtest_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate trading signals from predictions.

    Signal logic (direction-based):
    - signal = 1 (long) when predicted direction is UP (pred[i] > pred[i-1] + threshold)
    - signal = 0 (flat/cash) otherwise

    This uses PREDICTED DIRECTION rather than absolute price comparison because:
    1. Models may have systematic bias in absolute price levels (e.g., predicting
       lower than actual due to training on different price ranges)
    2. Direction prediction accuracy is often higher than level accuracy
    3. The threshold is applied to predicted change magnitude, not absolute price

    The signal_threshold parameter now controls the minimum predicted price
    increase required to generate a long signal (in price units, not percentage).

    Execution timing options:
    - "close": Signal at t means enter at close of day t, capture return t to t+1
    - "next_open": Signal at t means enter at open of day t+1, capture return t+1 to t+2

    Args:
        predictions: Dictionary containing predictions, actual, test_dates
        backtest_params: Dictionary containing:
            - signal_threshold: Min predicted increase for signal (default: 0)
            - initial_capital: Starting capital (default: 100000)
            - transaction_cost: Cost per trade as fraction (default: 0.001)
            - execution_timing: "close" or "next_open" (default: "close")

    Returns:
        Dictionary containing:
            - signals: Binary signals (0 or 1)
            - predictions: Predicted prices
            - actual: Actual prices
            - dates: Trading dates
            - params: Backtest parameters used
    """
    logger.info("Generating trading signals (direction-based)...")

    # Extract data
    pred = np.array(predictions["predictions"])
    actual = np.array(predictions["actual"])
    n_samples = len(pred)

    # Get parameters with defaults
    threshold = backtest_params.get("signal_threshold", 0.0)
    initial_capital = backtest_params.get("initial_capital", 100000)
    transaction_cost = backtest_params.get("transaction_cost", 0.001)
    execution_timing = backtest_params.get("execution_timing", "close")

    logger.info(f"Samples: {n_samples}")
    logger.info(f"Signal threshold (min predicted increase): {threshold}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    logger.info(f"Transaction cost: {transaction_cost*100:.2f}%")
    logger.info(f"Execution timing: {execution_timing}")

    # Validate execution_timing
    if execution_timing not in ("close", "next_open"):
        logger.warning(f"Unknown execution_timing '{execution_timing}', defaulting to 'close'")
        execution_timing = "close"

    # Generate signals based on PREDICTED DIRECTION (no look-ahead bias)
    # Signal[i] = 1 when pred[i] > pred[i-1] + threshold
    # This compares today's prediction with yesterday's prediction
    # If model predicts price will go UP, we go long
    signals = np.zeros(n_samples, dtype=int)

    # Start from index 1 since we need pred[i-1] for comparison
    for i in range(1, n_samples):
        # Compare prediction for day i with prediction for day i-1
        # If prediction is increasing (model predicts upward movement), go long
        if pred[i] > pred[i - 1] + threshold:
            signals[i] = 1
        else:
            signals[i] = 0

    # Count signal statistics
    long_count = np.sum(signals == 1)
    flat_count = np.sum(signals == 0)
    signal_changes = np.sum(np.abs(np.diff(signals)))

    logger.info(f"Long signals: {long_count} ({100*long_count/n_samples:.1f}%)")
    logger.info(f"Flat signals: {flat_count} ({100*flat_count/n_samples:.1f}%)")
    logger.info(f"Signal changes (trades): {signal_changes}")

    # Get dates if available
    dates = predictions.get("test_dates", list(range(n_samples)))

    # Get open prices if available (for next_open execution timing)
    open_prices = predictions.get("open_prices")
    if open_prices is not None:
        open_prices = np.array(open_prices).tolist()
        logger.info(f"Open prices available for backtest: {len(open_prices)} samples")
    else:
        logger.info("No open prices available - next_open timing will use close prices")

    return {
        "signals": signals.tolist(),
        "predictions": pred.tolist(),
        "actual": actual.tolist(),
        "open_prices": open_prices,  # For next_open execution timing
        "dates": dates,
        "n_samples": n_samples,
        "params": {
            "signal_threshold": threshold,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "execution_timing": execution_timing,
        },
        "signal_stats": {
            "long_count": int(long_count),
            "flat_count": int(flat_count),
            "signal_changes": int(signal_changes),
        },
    }


def run_backtest(
    signal_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Run backtest simulation and calculate performance metrics.

    Simulates a long-only strategy:
    - When signal=1: fully invested in the asset
    - When signal=0: in cash (0% return)

    Execution timing:
    - "close": Signal at t -> enter at close t, capture return from t to t+1
    - "next_open": Signal at t -> enter at open t+1, capture return from t+1 to t+2

    Args:
        signal_data: Dictionary containing signals, actual prices, dates, params

    Returns:
        Tuple of:
            - backtest_results: Full results including equity curve
            - backtest_metrics: Performance metrics as JSON
            - equity_curve: DataFrame with daily equity values
    """
    logger.info("Running backtest simulation...")

    # Extract data
    signals = np.array(signal_data["signals"])
    actual = np.array(signal_data["actual"])  # Close prices
    open_prices = signal_data.get("open_prices")
    if open_prices is not None:
        open_prices = np.array(open_prices)
    dates = signal_data["dates"]
    params = signal_data["params"]

    initial_capital = params["initial_capital"]
    transaction_cost = params["transaction_cost"]
    execution_timing = params.get("execution_timing", "close")
    n_samples = len(signals)

    logger.info(f"Execution timing: {execution_timing}")

    # Calculate daily returns of the asset (close-to-close)
    # Return[t] = (Close[t] - Close[t-1]) / Close[t-1]
    asset_returns = np.zeros(n_samples)
    asset_returns[1:] = np.diff(actual) / actual[:-1]

    # Calculate strategy returns based on execution timing
    strategy_returns = np.zeros(n_samples)

    if execution_timing == "next_open":
        # Signal at t -> enter at open of t+1, exit at close of t+1
        # Return = (Close[t+1] - Open[t+1]) / Open[t+1]
        if open_prices is not None:
            logger.info("Using actual Open prices for next_open execution")
            # Calculate open-to-close returns for each day
            open_to_close_returns = np.zeros(n_samples)
            for i in range(n_samples):
                if open_prices[i] > 0:  # Avoid division by zero
                    open_to_close_returns[i] = (actual[i] - open_prices[i]) / open_prices[i]

            for i in range(1, n_samples):
                if signals[i - 1] == 1:  # Use previous day's signal
                    strategy_returns[i] = open_to_close_returns[i]
                else:
                    strategy_returns[i] = 0.0

                # Apply transaction cost on signal change
                if i > 1 and signals[i - 1] != signals[i - 2]:
                    strategy_returns[i] -= transaction_cost
        else:
            # Fallback: shift close-to-close returns by 1 period
            logger.warning("No Open prices - using shifted close-to-close returns")
            for i in range(2, n_samples):
                if signals[i - 1] == 1:  # Use previous signal
                    strategy_returns[i] = asset_returns[i]
                else:
                    strategy_returns[i] = 0.0

                # Apply transaction cost on signal change
                if signals[i - 1] != signals[i - 2]:
                    strategy_returns[i] -= transaction_cost
    else:
        # "close" mode: Signal at t -> enter at close t, capture return t to t+1
        # CRITICAL FIX: Use previous day's signal to determine today's return
        # signals[i-1] was generated at close of day i-1, so we can trade at close i-1
        # and capture return from close i-1 to close i (which is asset_returns[i])
        for i in range(1, n_samples):
            if signals[i - 1] == 1:  # Previous day's signal
                # We went long at previous close, capture today's return
                strategy_returns[i] = asset_returns[i]
            else:
                # We were flat (in cash), no return
                strategy_returns[i] = 0.0

            # Apply transaction cost on signal change
            if i > 1 and signals[i - 1] != signals[i - 2]:
                strategy_returns[i] -= transaction_cost

    # Calculate cumulative returns
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    buyhold_cumulative = np.cumprod(1 + asset_returns)

    # Calculate equity curves
    strategy_equity = initial_capital * strategy_cumulative
    buyhold_equity = initial_capital * buyhold_cumulative

    logger.info(f"Strategy final equity: ${strategy_equity[-1]:,.2f}")
    logger.info(f"Buy & Hold final equity: ${buyhold_equity[-1]:,.2f}")

    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        strategy_returns=strategy_returns,
        asset_returns=asset_returns,
        strategy_equity=strategy_equity,
        buyhold_equity=buyhold_equity,
        initial_capital=initial_capital,
        signals=signals,
    )

    # Log key metrics
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    logger.info("=" * 60)
    logger.info(f"Buy & Hold Return: {metrics['buyhold_return']*100:.2f}%")
    logger.info(
        f"Outperformance: {(metrics['total_return'] - metrics['buyhold_return'])*100:.2f}%"
    )
    logger.info("=" * 60)

    # Create equity curve DataFrame
    # Ensure dates has the same length as other arrays
    if len(dates) != n_samples:
        logger.warning(f"Dates length ({len(dates)}) != samples ({n_samples}), using indices")
        dates = list(range(n_samples))

    equity_df = pd.DataFrame(
        {
            "date": dates,
            "signal": signals,
            "actual_price": actual,
            "strategy_return": strategy_returns,
            "asset_return": asset_returns,
            "strategy_equity": strategy_equity,
            "buyhold_equity": buyhold_equity,
        }
    )

    # Prepare full results
    backtest_results = {
        "signals": signals.tolist(),
        "strategy_returns": strategy_returns.tolist(),
        "asset_returns": asset_returns.tolist(),
        "strategy_equity": strategy_equity.tolist(),
        "buyhold_equity": buyhold_equity.tolist(),
        "dates": dates,
        "params": params,
        "metrics": metrics,
    }

    # Prepare JSON-serializable metrics
    backtest_metrics = {
        "total_return": metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "volatility": metrics["volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar_ratio": metrics["calmar_ratio"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "total_trades": metrics["total_trades"],
        "avg_trade_return": metrics["avg_trade_return"],
        "buyhold_return": metrics["buyhold_return"],
        "outperformance": metrics["total_return"] - metrics["buyhold_return"],
        "n_samples": n_samples,
        "initial_capital": initial_capital,
        "final_equity": float(strategy_equity[-1]),
    }

    return backtest_results, backtest_metrics, equity_df


def calculate_performance_metrics(
    strategy_returns: np.ndarray,
    asset_returns: np.ndarray,
    strategy_equity: np.ndarray,
    buyhold_equity: np.ndarray,
    initial_capital: float,
    signals: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics.

    Args:
        strategy_returns: Daily strategy returns
        asset_returns: Daily asset returns
        strategy_equity: Strategy equity curve
        buyhold_equity: Buy & Hold equity curve
        initial_capital: Starting capital
        signals: Trading signals

    Returns:
        Dictionary of performance metrics
    """
    n = len(strategy_returns)

    # Total Return
    total_return = (strategy_equity[-1] - initial_capital) / initial_capital

    # Annualized Return
    years = n / TRADING_DAYS_PER_YEAR
    if years > 0:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0.0

    # Volatility (annualized)
    daily_vol = np.std(strategy_returns)
    volatility = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe Ratio (assuming risk-free rate = 0)
    if volatility > 0:
        sharpe_ratio = annual_return / volatility
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown
    peak = np.maximum.accumulate(strategy_equity)
    drawdown = (strategy_equity - peak) / peak
    max_drawdown = np.min(drawdown)

    # Calmar Ratio
    if max_drawdown < 0:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = float("inf") if annual_return > 0 else 0.0

    # Win Rate (percentage of positive return days when invested)
    invested_returns = strategy_returns[signals == 1]
    if len(invested_returns) > 0:
        win_rate = np.mean(invested_returns > 0)
    else:
        win_rate = 0.0

    # Profit Factor (sum of gains / sum of losses)
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    if losses > 0:
        profit_factor = gains / losses
    else:
        profit_factor = float("inf") if gains > 0 else 0.0

    # Total trades (signal changes)
    total_trades = int(np.sum(np.abs(np.diff(signals))))

    # Average trade return
    if total_trades > 0:
        avg_trade_return = total_return / total_trades
    else:
        avg_trade_return = 0.0

    # Buy & Hold Return
    buyhold_return = (buyhold_equity[-1] - initial_capital) / initial_capital

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe_ratio),
        "volatility": float(volatility),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "total_trades": total_trades,
        "avg_trade_return": float(avg_trade_return),
        "buyhold_return": float(buyhold_return),
    }
