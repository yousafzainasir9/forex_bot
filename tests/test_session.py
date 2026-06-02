"""Tests for the trading-session window helper and the backtest entry gates
(session filter + higher-timeframe alignment). All offline / deterministic."""

from bot.config import in_trading_session
from bot.backtest import run_backtest, make_synthetic, StrategyParams


# --------------------------- pure session helper -------------------------
def test_session_normal_window():
    # 07:00–16:00 UTC: open at 7, closed at 16 (exclusive) and before 7.
    assert not in_trading_session(6, 7, 16)
    assert in_trading_session(7, 7, 16)
    assert in_trading_session(15, 7, 16)
    assert not in_trading_session(16, 7, 16)
    assert not in_trading_session(20, 7, 16)


def test_session_wraps_midnight():
    # 22:00–06:00 UTC window spans midnight.
    assert in_trading_session(23, 22, 6)
    assert in_trading_session(0, 22, 6)
    assert in_trading_session(5, 22, 6)
    assert not in_trading_session(6, 22, 6)
    assert not in_trading_session(12, 22, 6)


def test_session_equal_bounds_is_always_open():
    for h in range(24):
        assert in_trading_session(h, 0, 0)


# --------------------------- backtest entry gates ------------------------
def test_session_filter_reduces_or_equals_trades():
    df = make_synthetic(n=3000, seed=3)
    base = StrategyParams()  # legacy filter mode -> produces some trades
    unfiltered = run_backtest(df, base, session_filter=False).summary["n_trades"]
    # A narrow 1-hour window can only remove entries, never add them.
    filtered = run_backtest(
        df, base, session_filter=True, session_start_hour=8, session_end_hour=9
    ).summary["n_trades"]
    assert filtered <= unfiltered


def test_htf_gate_reduces_or_equals_trades():
    df = make_synthetic(n=3000, seed=3)
    base = StrategyParams()
    ungated = run_backtest(df, base, require_htf_align=False).summary["n_trades"]
    gated = run_backtest(
        df, base, require_htf_align=True, htf_timeframe="M15"
    ).summary["n_trades"]
    # The gate can only filter trades out, so it never increases the count.
    assert gated <= ungated
