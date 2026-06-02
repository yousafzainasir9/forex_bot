"""Unit tests for the indicator engine — verified against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from bot.indicators import add_indicators, atr, ema, rsi, true_range


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def test_ema_recursive_matches_manual():
    # EMA with span=3 → alpha = 2/(3+1) = 0.5, adjust=False, seeded at the 3rd point.
    s = _series([1, 2, 3, 4, 5])
    out = ema(s, 3)
    # First two are NaN (min_periods=3). Seed at index 2 = mean(1,2,3)? No —
    # pandas ewm(adjust=False) seeds with the first value, but min_periods masks
    # outputs until enough points. Value at idx2 = recursive from first obs.
    # Recurrence: e0=1, e1=0.5*2+0.5*1=1.5, e2=0.5*3+0.5*1.5=2.25
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.25)
    # e3 = 0.5*4 + 0.5*2.25 = 3.125 ; e4 = 0.5*5 + 0.5*3.125 = 4.0625
    assert out.iloc[3] == pytest.approx(3.125)
    assert out.iloc[4] == pytest.approx(4.0625)


def test_ema_constant_series_equals_constant():
    s = _series([5.0] * 30)
    out = ema(s, 9)
    assert out.dropna().eq(5.0).all()


def test_rsi_all_gains_is_100():
    s = _series(list(range(1, 30)))  # strictly increasing → no losses
    out = rsi(s, 14)
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_range_bounds():
    rng = np.random.default_rng(0)
    s = _series(100 + np.cumsum(rng.normal(0, 1, 200)))
    out = rsi(s, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_rsi_known_wilder_value():
    # Classic Wilder example sequence (close prices).
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
    s = _series(closes)
    out = rsi(s, 14)
    # Wilder's first RSI for this canonical series ≈ 70.46.
    assert out.iloc[14] == pytest.approx(70.46, abs=0.5)


def test_true_range_and_atr_positive():
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    tr = true_range(df)
    a = atr(df, 14)
    assert (tr.dropna() >= 0).all()
    assert (a.dropna() > 0).all()
    assert np.isnan(a.iloc[0])  # warmup


def test_add_indicators_no_lookahead():
    # Truncating the series must not change earlier indicator values (causality).
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(2)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + 0.0002
    low = close - 0.0002
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    full = add_indicators(df, ema_fast=9, ema_slow=21, rsi_period=14, atr_period=14)
    truncated = add_indicators(df.iloc[:60], ema_fast=9, ema_slow=21,
                               rsi_period=14, atr_period=14)
    for col in ["ema_fast", "ema_slow", "rsi", "atr"]:
        a = full[col].iloc[:60].dropna()
        b = truncated[col].dropna()
        common = a.index.intersection(b.index)
        assert np.allclose(a.loc[common].values, b.loc[common].values, atol=1e-9)
