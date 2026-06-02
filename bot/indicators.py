"""Module 3 — Indicator Engine.

Pure, dependency-light indicator functions computed directly from candle data.
No TA library is required; values are verified against hand-computed references
in tests/test_indicators.py.

CAUSALITY GUARANTEE
-------------------
Every value at bar ``t`` uses only information available up to and including bar
``t`` (no look-ahead). We never use ``.shift(-1)``, future rows, or centered
windows. This is the single most important property for an honest backtest.

Conventions match the standard MT5 / Wilder definitions:
  * EMA  — exponential moving average, adjust=False (recursive, like MT5).
  * RSI  — Wilder's smoothing (RMA), period default 14.
  * ATR  — Wilder's smoothing of True Range, period default 14.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (recursive, MT5-style, no look-ahead).

    Uses ``adjust=False`` so each value depends only on the prior EMA and the
    current price — identical to the recurrence MT5 uses on its charts.
    """
    if period <= 0:
        raise ValueError("EMA period must be positive.")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average (a.k.a. RMA / SMMA).

    Equivalent to an EMA with alpha = 1/period. Used by both RSI and ATR.
    The first value is seeded with a simple average of the first ``period``
    observations (Wilder's original method), which matches MT5.
    """
    arr = series.to_numpy(dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype="float64")
    if n < period:
        return pd.Series(out, index=series.index)

    # Seed: simple mean of the first `period` values.
    seed = np.nanmean(arr[:period])
    out[period - 1] = seed
    alpha = 1.0 / period
    for i in range(period, n):
        prev = out[i - 1]
        out[i] = prev + alpha * (arr[i] - prev)
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (matches MT5).

    Returns values in [0, 100]. The first ``period`` rows are NaN.
    """
    if period <= 0:
        raise ValueError("RSI period must be positive.")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    # Avoid divide-by-zero: where avg_loss == 0 → RSI = 100.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0.0] = 100.0
    out[(avg_gain == 0.0) & (avg_loss == 0.0)] = 50.0
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (matches MT5)."""
    if period <= 0:
        raise ValueError("ATR period must be positive.")
    tr = true_range(df)
    return _wilder_rma(tr, period)


def add_indicators(
    df: pd.DataFrame,
    *,
    ema_fast: int = 9,
    ema_slow: int = 21,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Return a copy of ``df`` with indicator columns appended.

    Expects columns: open, high, low, close (volume optional). Index should be a
    UTC DatetimeIndex of *closed* candles. Adds:
      ema_fast, ema_slow, rsi, atr.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out, atr_period)
    return out
