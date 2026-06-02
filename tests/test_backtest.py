"""Tests for the walk-forward backtester engine.

Signal generation is stubbed in several tests so we can verify the ENGINE
mechanics (entry timing, SL/TP fills, costs, daily halt) deterministically,
independent of indicator math (which is tested in test_indicators/test_strategy).
"""

import numpy as np
import pandas as pd
import pytest

import bot.backtest as bt
from bot.backtest import (CostModel, StrategyParams, run_backtest, walk_forward,
                          load_csv, make_synthetic, _summarize)
from bot.strategy import Action, PositionSide, Signal


def _flat_frame(n=60, price=1.1000, hi=1.1001, lo=1.0999):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": [price]*n, "high": [hi]*n, "low": [lo]*n,
                         "close": [price]*n}, index=idx)


def _buy_once_stub(trigger_i):
    """Return an evaluate() replacement that BUYs only at bar index trigger_i."""
    def fake(win, *, rsi_overbought=70.0, rsi_oversold=30.0, open_side=None):
        row = win.iloc[-1]
        is_trigger = (len(win) and win.attrs.get("_i") == trigger_i)
        action = Action.BUY if (open_side is None and is_trigger) else Action.HOLD
        return Signal(action=action, reason="stub", price=float(row["close"]),
                      atr=0.0010, rsi=50.0, ema_fast=1.1, ema_slow=1.0,
                      bar_time=win.index[-1])
    return fake


def _run_with_stub(df, stub, **kw):
    """Patch evaluate inside backtest and tag each window with its bar index."""
    import bot.backtest as mod
    orig = mod.evaluate
    # Wrap so the stub can know which bar index it's on via win.attrs.
    enriched_holder = {}
    def wrapper(win, **kwargs):
        # infer index position from the last timestamp against the full frame
        return stub(win, **kwargs)
    mod.evaluate = wrapper
    try:
        return run_backtest(df, **kw)
    finally:
        mod.evaluate = orig


# --- Engine mechanics via a tagged-index stub --------------------------------

def _run_forced(df, trigger_i, **kw):
    """Run the engine but force a BUY at one specific bar index."""
    import bot.backtest as mod
    orig = mod.evaluate
    def fake(win, *, rsi_overbought=70.0, rsi_oversold=30.0, open_side=None):
        row = win.iloc[-1]
        cur_i = df.index.get_loc(win.index[-1])
        action = Action.BUY if (open_side is None and cur_i == trigger_i) else Action.HOLD
        return Signal(action=action, reason="stub", price=float(row["close"]),
                      atr=0.0010, rsi=50.0, ema_fast=1.1, ema_slow=1.0,
                      bar_time=win.index[-1])
    mod.evaluate = fake
    try:
        return run_backtest(df, **kw)
    finally:
        mod.evaluate = orig


def test_take_profit_exit_and_next_open_entry():
    df = _flat_frame(60)
    # Spike up at bar 32 so the long (entered at bar 31 open) hits TP.
    df.iloc[32, df.columns.get_loc("high")] = 1.1100
    res = _run_forced(df, trigger_i=30, costs=CostModel(spread_price=0.0, commission_per_lot=0.0))
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["reason_close"] == "TP"
    assert t["entry"] == pytest.approx(1.1000, abs=1e-9)   # next-bar (31) open, no spread
    assert t["pnl"] > 0
    assert t["r_multiple"] == pytest.approx(1.5, abs=0.05)  # TP sits at 1.5R


def test_stop_loss_exit_is_negative_one_R():
    df = _flat_frame(60)
    df.iloc[32, df.columns.get_loc("low")] = 1.0900   # crash down → hit SL
    res = _run_forced(df, trigger_i=30, costs=CostModel(spread_price=0.0, commission_per_lot=0.0))
    t = res.trades.iloc[0]
    assert t["reason_close"] == "SL"
    assert t["pnl"] < 0
    assert t["r_multiple"] == pytest.approx(-1.0, abs=0.05)


def test_commission_reduces_pnl():
    df = _flat_frame(60)
    df.iloc[32, df.columns.get_loc("high")] = 1.1100
    free = _run_forced(df, 30, costs=CostModel(spread_price=0.0, commission_per_lot=0.0))
    paid = _run_forced(df, 30, costs=CostModel(spread_price=0.0, commission_per_lot=20.0))
    lots = free.trades.iloc[0]["lots"]
    assert paid.trades.iloc[0]["pnl"] == pytest.approx(
        free.trades.iloc[0]["pnl"] - 20.0 * lots, abs=1e-6)


def test_spread_worsens_entry():
    df = _flat_frame(60)
    df.iloc[32, df.columns.get_loc("high")] = 1.1100
    res = _run_forced(df, 30, costs=CostModel(spread_price=0.0002, commission_per_lot=0.0))
    # long entry pays half... here full spread on the adverse side: open + spread
    assert res.trades.iloc[0]["entry"] == pytest.approx(1.1002, abs=1e-9)


def test_no_trade_when_flat_and_no_signal():
    df = _flat_frame(60)
    res = _run_forced(df, trigger_i=-1)  # never triggers
    assert len(res.trades) == 0
    assert res.summary["n_trades"] == 0
    assert res.summary["final_equity"] == 10_000.0


# --- Summary maths -----------------------------------------------------------

def test_summary_invariants_on_synthetic():
    df = make_synthetic(900, seed=3)
    res = run_backtest(df)
    s = res.summary
    assert s["wins"] + s["losses"] == s["n_trades"]
    if s["n_trades"]:
        assert s["final_equity"] == pytest.approx(10_000.0 + s["net_pnl"], abs=1.0)


def test_summarize_empty():
    s = _summarize([], 10_000.0)
    assert s["n_trades"] == 0 and s["final_equity"] == 10_000.0


# --- Walk-forward ------------------------------------------------------------

def test_walk_forward_produces_oos():
    df = make_synthetic(1500, seed=11)
    wf = walk_forward(df, is_bars=700, oos_bars=300)
    assert len(wf.folds) >= 2
    assert set(["n_trades", "net_pnl", "profit_factor"]).issubset(wf.oos_summary)
    for f in wf.folds:
        assert f["params"]["ema_fast"] < f["params"]["ema_slow"]  # never invalid


def test_walk_forward_deterministic():
    df = make_synthetic(1300, seed=5)
    a = walk_forward(df, is_bars=600, oos_bars=300).oos_summary
    b = walk_forward(df, is_bars=600, oos_bars=300).oos_summary
    assert a == b


# --- Data loading ------------------------------------------------------------

def test_make_synthetic_deterministic_shape():
    a = make_synthetic(500, seed=1)
    b = make_synthetic(500, seed=1)
    assert a.equals(b)
    assert list(a.columns) == ["open", "high", "low", "close"]
    assert (a["high"] >= a["low"]).all()


def test_load_csv_roundtrip(tmp_path):
    df = make_synthetic(50, seed=2).reset_index().rename(columns={"index": "time"})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    loaded = load_csv(p)
    assert list(loaded.columns)[:4] == ["open", "high", "low", "close"]
    assert len(loaded) == 50


def test_load_csv_missing_cols(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"time": ["2025-01-01"], "open": [1.1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_csv(p)


# --- Grid-search heatmap + HTML report ---------------------------------------

def test_grid_search_shape_and_invalids():
    from bot.backtest import grid_search
    df = make_synthetic(700, seed=4)
    hm = grid_search(df, "ema_fast", [9, 30], "ema_slow", [21, 8], metric="net_pnl")
    assert hm["x_param"] == "ema_fast" and hm["y_param"] == "ema_slow"
    assert len(hm["matrix"]) == 2 and len(hm["matrix"][0]) == 2
    # ema_slow=8 row: fast=9>=8 invalid -> None; fast=30>=8 invalid -> None
    assert hm["matrix"][1] == [None, None]
    # ema_slow=21 row: fast=9<21 valid (number), fast=30>=21 invalid -> None
    assert isinstance(hm["matrix"][0][0], (int, float)) and hm["matrix"][0][1] is None


def test_build_backtest_html_valid():
    from bot.backtest import build_backtest_html, grid_search
    df = make_synthetic(800, seed=6)
    res = run_backtest(df)
    hm = grid_search(df, "ema_fast", [9, 12], "ema_slow", [21, 55], metric="net_pnl")
    html = build_backtest_html(res, hm, title="T")
    assert "{cards}" not in html and "{trades_tbl}" not in html  # all f-substituted
    assert 'id="eq"' in html and 'class="heat"' in html
    assert "chart.umd.min.js" in html


def test_build_backtest_html_no_trades():
    from bot.backtest import build_backtest_html
    # A frame too short to trade -> empty result still renders.
    df = make_synthetic(40, seed=1)
    res = run_backtest(df)
    html = build_backtest_html(res, None)
    assert "<html" in html and "Backtest report" in html
