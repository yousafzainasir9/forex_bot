"""Unit tests for the multi-symbol opportunity scanner.

Covers the PURE scorer (score_opportunity), ranking, and scan_symbol against a
fake in-memory feed so nothing here touches MT5 or the network.
"""

import numpy as np
import pandas as pd
import pytest

from bot.config import Settings, _parse_symbols, DEFAULT_WATCHLIST
from bot.scanner import (
    ScanScore, SymbolScan, score_opportunity, rank_opportunities,
    scan_symbol, scan_watchlist, SPREAD_SKIP_FRAC,
)
from bot.strategy import Action


# --------------------------- pure scorer ---------------------------------
def _score(**kw):
    base = dict(action=Action.BUY, ema_fast=1.1010, ema_slow=1.1000, atr=0.0010,
                price=1.1000, rsi=60.0, spread=0.00005, atr_multiplier=1.5,
                htf_trend="UP")
    base.update(kw)
    return score_opportunity(**base)


def test_score_in_unit_range_and_tradable():
    s = _score()
    assert s.tradable
    assert 0.0 < s.total <= 1.0
    for c in (s.signal, s.volatility, s.trend, s.spread):
        assert 0.0 <= c <= 1.0


def test_hold_is_not_tradable():
    s = _score(action=Action.HOLD)
    assert not s.tradable and "no entry" in s.skip_reason


def test_htf_gate_blocks_counter_trend_long():
    # BUY against a DOWN higher-timeframe trend is gated out when alignment required.
    s = _score(action=Action.BUY, htf_trend="DOWN", require_htf_align=True)
    assert not s.tradable and "gated" in s.skip_reason


def test_htf_gate_allows_aligned_long():
    s = _score(action=Action.BUY, htf_trend="UP", require_htf_align=True)
    assert s.tradable


def test_htf_gate_blocks_unknown_trend():
    s = _score(action=Action.BUY, htf_trend=None, require_htf_align=True)
    assert not s.tradable and "unknown" in s.skip_reason


def test_htf_gate_off_allows_counter_trend():
    # Without the gate, a counter-trend signal is still tradable (just scored lower).
    s = _score(action=Action.BUY, htf_trend="DOWN", require_htf_align=False)
    assert s.tradable


def test_zero_atr_not_tradable():
    s = _score(atr=0.0)
    assert not s.tradable and "atr" in s.skip_reason.lower()


def test_wide_spread_is_skipped():
    # spread bigger than SPREAD_SKIP_FRAC of the stop distance (0.0010*1.5=0.0015)
    s = _score(spread=0.0015 * (SPREAD_SKIP_FRAC + 0.2))
    assert not s.tradable and "spread" in s.skip_reason.lower()


def test_trend_alignment_rewards_and_punishes():
    aligned = _score(action=Action.BUY, htf_trend="UP")
    opposed = _score(action=Action.BUY, htf_trend="DOWN")
    assert aligned.trend == 1.0
    assert opposed.trend == 0.0
    assert aligned.total > opposed.total


def test_unknown_trend_is_neutral():
    s = _score(htf_trend=None)
    assert s.trend == 0.5


def test_sell_rsi_momentum_counts_downside():
    # For a SELL, low RSI = strong downside momentum -> higher signal component.
    strong = _score(action=Action.SELL, rsi=25.0, htf_trend="DOWN")
    weak = _score(action=Action.SELL, rsi=55.0, htf_trend="DOWN")
    assert strong.signal > weak.signal


def test_higher_volatility_scores_higher():
    lo = _score(atr=0.0003)
    hi = _score(atr=0.0020)
    assert hi.volatility >= lo.volatility


# --------------------------- ranking -------------------------------------
def test_rank_sorts_tradable_by_score_desc():
    a = SymbolScan("AAA", True, action="BUY", score=0.4, tradable=True)
    b = SymbolScan("BBB", True, action="SELL", score=0.8, tradable=True)
    c = SymbolScan("CCC", True, action="HOLD", score=0.9, tradable=False)
    ranked = rank_opportunities([a, b, c])
    assert [r.symbol for r in ranked] == ["BBB", "AAA"]  # C excluded (not tradable)


# --------------------------- fake-feed integration -----------------------
def _bullish_cross_df(n=120):
    """Synthetic OHLC where a fresh bullish EMA cross prints on the last bar."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # downtrend then sharp up-move at the end to force fast EMA above slow EMA.
    base = np.linspace(1.10, 1.09, n - 10).tolist() + np.linspace(1.09, 1.105, 10).tolist()
    close = pd.Series(base, index=idx)
    high = close + 0.0003
    low = close - 0.0003
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]),
                         "high": high, "low": low, "close": close,
                         "volume": 100}, index=idx)


class _Tick:
    def __init__(self, spread): self.spread = spread


class FakeFeed:
    def __init__(self, df, available=True, spread=0.00005):
        self._df, self._available, self._spread = df, available, spread

    def ensure_symbol(self, symbol):
        return self._available

    def get_candles(self, symbol=None, timeframe=None, n=None):
        return self._df.tail(n) if n else self._df

    def get_tick(self, symbol=None):
        return _Tick(self._spread)


def _settings():
    return Settings.load()


def test_scan_symbol_unavailable_is_skipped():
    s = _settings()
    res = scan_symbol(FakeFeed(_bullish_cross_df(), available=False), s, "ZZZ")
    assert not res.available and not res.tradable


def test_scan_symbol_produces_scored_result():
    s = _settings()
    res = scan_symbol(FakeFeed(_bullish_cross_df()), s, "EURUSD")
    assert res.available
    assert res.action in ("BUY", "SELL", "HOLD", "CLOSE")
    # JSON shape sane for the web app
    d = res.to_dict()
    assert d["symbol"] == "EURUSD" and "components" in d


def test_scan_watchlist_runs_all_symbols():
    s = _settings()
    feed = FakeFeed(_bullish_cross_df())
    # restrict to 3 symbols for speed
    import dataclasses
    s = dataclasses.replace(s, symbols=["EURUSD", "GBPUSD", "XAUUSD"])
    scans = scan_watchlist(feed, s)
    assert len(scans) == 3
    assert {x.symbol for x in scans} == {"EURUSD", "GBPUSD", "XAUUSD"}


# --------------------------- config parsing ------------------------------
def test_default_watchlist_used_when_unset(monkeypatch):
    monkeypatch.delenv("SYMBOLS", raising=False)
    assert _parse_symbols(None, "EURUSD") == list(DEFAULT_WATCHLIST)


def test_single_keyword_and_csv():
    assert _parse_symbols("SINGLE", "EURUSD") == ["EURUSD"]
    assert _parse_symbols("eurusd, gbpusd; xauusd", "EURUSD") == ["EURUSD", "GBPUSD", "XAUUSD"]
