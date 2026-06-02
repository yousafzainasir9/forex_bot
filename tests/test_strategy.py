"""Unit tests for the strategy engine against synthetic crossover sequences."""

import numpy as np
import pandas as pd

from bot.strategy import Action, PositionSide, evaluate


def _frame(rows):
    """Build a minimal indicator-enriched frame from explicit indicator rows.

    Each row: (ema_fast, ema_slow, rsi). close/atr filled with sane constants.
    """
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    data = {
        "open": [1.10] * len(rows),
        "high": [1.11] * len(rows),
        "low": [1.09] * len(rows),
        "close": [1.10] * len(rows),
        "atr": [0.0010] * len(rows),
        "ema_fast": [r[0] for r in rows],
        "ema_slow": [r[1] for r in rows],
        "rsi": [r[2] for r in rows],
    }
    return pd.DataFrame(data, index=idx)


def test_bullish_cross_with_ok_rsi_is_buy():
    df = _frame([(0.9, 1.0, 50), (1.1, 1.0, 55)])  # fast crosses above slow
    sig = evaluate(df)
    assert sig.action is Action.BUY


def test_bullish_cross_but_overbought_is_hold():
    df = _frame([(0.9, 1.0, 75), (1.1, 1.0, 75)])  # RSI >= 70 blocks the long
    sig = evaluate(df)
    assert sig.action is Action.HOLD
    assert "overbought" in sig.reason


def test_bearish_cross_with_ok_rsi_is_sell():
    df = _frame([(1.1, 1.0, 50), (0.9, 1.0, 45)])  # fast crosses below slow
    sig = evaluate(df)
    assert sig.action is Action.SELL


def test_bearish_cross_but_oversold_is_hold():
    df = _frame([(1.1, 1.0, 25), (0.9, 1.0, 25)])  # RSI <= 30 blocks the short
    sig = evaluate(df)
    assert sig.action is Action.HOLD
    assert "oversold" in sig.reason


def test_no_cross_is_hold():
    df = _frame([(1.1, 1.0, 50), (1.2, 1.0, 50)])  # fast stays above, no cross
    sig = evaluate(df)
    assert sig.action is Action.HOLD


def test_opposite_cross_closes_long():
    df = _frame([(1.1, 1.0, 50), (0.9, 1.0, 45)])  # bearish cross
    sig = evaluate(df, open_side=PositionSide.LONG)
    assert sig.action is Action.CLOSE


def test_opposite_cross_closes_short():
    df = _frame([(0.9, 1.0, 50), (1.1, 1.0, 55)])  # bullish cross
    sig = evaluate(df, open_side=PositionSide.SHORT)
    assert sig.action is Action.CLOSE


def test_open_position_no_opposite_cross_holds():
    df = _frame([(1.2, 1.0, 50), (1.3, 1.0, 50)])  # still long-biased
    sig = evaluate(df, open_side=PositionSide.LONG)
    assert sig.action is Action.HOLD


def test_warmup_nan_is_hold():
    idx = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [1.1, 1.1], "high": [1.1, 1.1], "low": [1.1, 1.1], "close": [1.1, 1.1],
        "atr": [np.nan, np.nan], "ema_fast": [np.nan, 1.1],
        "ema_slow": [np.nan, 1.0], "rsi": [np.nan, 50],
    }, index=idx)
    sig = evaluate(df)
    assert sig.action is Action.HOLD
