"""Module 8 — Opportunity Scanner (multi-symbol).

Turns the single-pair bot into a watchlist scanner. Each cycle it walks every
symbol in ``settings.symbols``, runs the SAME M5 EMA/RSI/ATR strategy, and scores
how attractive the opportunity is so the runner can trade the strongest one(s).

The scoring is deliberately TRANSPARENT and PURE (``score_opportunity`` takes
plain numbers and returns a breakdown), so it is fully unit-testable and you can
see exactly why one symbol ranked above another.

Ranking blends four industry-standard ideas:
  * signal strength   — how decisively the strategy fired (EMA separation vs ATR,
                        and RSI momentum in the trade's direction).
  * volatility (ATR%) — ATR as a fraction of price; favors symbols actually
                        moving enough for the target to be reachable.
  * trend alignment   — does the M5 entry agree with the higher-timeframe (e.g.
                        M15) EMA trend? Multi-timeframe confirmation.
  * spread cost       — fraction of the planned stop distance eaten by the live
                        spread; wide-spread instruments are penalised, and an
                        absurd spread (> SPREAD_SKIP_FRAC of the stop) is skipped.

NOTHING here places orders or talks to MT5 directly except through the injected
``feed``/``executor`` interfaces, so the maths stays testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .indicators import add_indicators
from .strategy import Action, evaluate


# Default blend weights (sum to 1.0). Tunable; exposed for tests/experiments.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "signal": 0.40,
    "volatility": 0.20,
    "trend": 0.25,
    "spread": 0.15,
}

# ATR% (atr/price) that counts as "fully volatile enough" -> volatility score 1.0.
# 0.0012 ~ 12 bps per M5 bar; gold/indices/crypto easily exceed it, majors sit below.
VOL_TARGET_ATR_PCT = 0.0012

# Spread as a fraction of the planned stop distance (atr * atr_multiplier):
#   <= GOOD            -> no penalty
#   GOOD..SKIP         -> linear penalty
#   >  SKIP            -> opportunity is dropped (spread eats too much edge)
SPREAD_GOOD_FRAC = 0.10
SPREAD_SKIP_FRAC = 0.50


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ScanScore:
    """Pure scoring breakdown for one symbol's opportunity."""
    total: float
    signal: float
    volatility: float
    trend: float
    spread: float
    tradable: bool
    skip_reason: str = ""


def score_opportunity(
    *,
    action: Action,
    ema_fast: float,
    ema_slow: float,
    atr: float,
    price: float,
    rsi: float,
    spread: float,
    atr_multiplier: float,
    htf_trend: Optional[str],   # "UP" / "DOWN" / None (unknown)
    weights: Optional[Dict[str, float]] = None,
) -> ScanScore:
    """Score a single opportunity from plain numbers. Pure & deterministic.

    Returns a ``ScanScore`` with each [0,1] component and the weighted total.
    ``tradable`` is False (with a reason) for non-actionable signals, degenerate
    inputs, or a spread that eats more than ``SPREAD_SKIP_FRAC`` of the stop.
    """
    w = weights or DEFAULT_WEIGHTS

    # Only BUY / SELL are tradable opportunities.
    if action not in (Action.BUY, Action.SELL):
        return ScanScore(0.0, 0.0, 0.0, 0.0, 0.0, False, f"{action.value} (no entry)")
    if not (atr and atr > 0) or not (price and price > 0):
        return ScanScore(0.0, 0.0, 0.0, 0.0, 0.0, False, "invalid atr/price")

    stop_distance = atr * atr_multiplier

    # --- spread gate + component ---
    spread = max(0.0, float(spread or 0.0))
    spread_frac = spread / stop_distance if stop_distance > 0 else 1.0
    if spread_frac > SPREAD_SKIP_FRAC:
        return ScanScore(0.0, 0.0, 0.0, 0.0, 0.0, False,
                         f"spread {spread_frac:.0%} of stop > {SPREAD_SKIP_FRAC:.0%}")
    if spread_frac <= SPREAD_GOOD_FRAC:
        spread_c = 1.0
    else:
        spread_c = _clamp(1.0 - (spread_frac - SPREAD_GOOD_FRAC) /
                          (SPREAD_SKIP_FRAC - SPREAD_GOOD_FRAC))

    # --- signal strength: EMA decisiveness (vs ATR) + RSI momentum in-direction ---
    ema_decisiveness = _clamp(abs(ema_fast - ema_slow) / atr) if atr > 0 else 0.0
    if action is Action.BUY:
        rsi_mom = _clamp((rsi - 50.0) / 30.0)
    else:
        rsi_mom = _clamp((50.0 - rsi) / 30.0)
    signal_c = _clamp(0.6 * ema_decisiveness + 0.4 * rsi_mom)

    # --- volatility (ATR%) ---
    atr_pct = atr / price
    vol_c = _clamp(atr_pct / VOL_TARGET_ATR_PCT)

    # --- higher-timeframe trend alignment ---
    if htf_trend is None:
        trend_c = 0.5  # unknown -> neutral, neither rewarded nor punished
    else:
        want = "UP" if action is Action.BUY else "DOWN"
        trend_c = 1.0 if htf_trend == want else 0.0

    total = (w["signal"] * signal_c + w["volatility"] * vol_c
             + w["trend"] * trend_c + w["spread"] * spread_c)
    return ScanScore(
        total=round(total, 4),
        signal=round(signal_c, 4),
        volatility=round(vol_c, 4),
        trend=round(trend_c, 4),
        spread=round(spread_c, 4),
        tradable=True,
    )


@dataclass
class SymbolScan:
    """Per-symbol scan result the runner ranks and the web app renders."""
    symbol: str
    available: bool
    action: str = "HOLD"
    reason: str = ""
    price: float = 0.0
    rsi: float = float("nan")
    atr: float = float("nan")
    atr_pct: float = 0.0
    spread: float = 0.0
    spread_frac: float = 0.0
    htf_trend: Optional[str] = None
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    tradable: bool = False
    skip_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "action": self.action,
            "reason": self.reason,
            "price": round(self.price, 6) if self.price else 0.0,
            "rsi": (None if (self.rsi != self.rsi) else round(self.rsi, 1)),
            "atr": (None if (self.atr != self.atr) else round(self.atr, 6)),
            "atr_pct": round(self.atr_pct * 100, 3),   # percent
            "spread": round(self.spread, 6),
            "spread_frac": round(self.spread_frac, 4),
            "htf_trend": self.htf_trend,
            "score": round(self.score, 4),
            "components": self.components,
            "tradable": self.tradable,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


def _htf_trend(feed, settings, symbol: str) -> Optional[str]:
    """Higher-timeframe EMA trend for ``symbol``: 'UP', 'DOWN', or None on failure."""
    try:
        need = settings.ema_slow + settings.atr_period + 5
        htf = feed.get_candles(symbol=symbol, timeframe=settings.htf_timeframe, n=max(need, 80))
        if htf is None or len(htf) < settings.ema_slow + 1:
            return None
        enr = add_indicators(htf, ema_fast=settings.ema_fast, ema_slow=settings.ema_slow,
                             rsi_period=settings.rsi_period, atr_period=settings.atr_period)
        last = enr.iloc[-1]
        if pd.isna(last["ema_fast"]) or pd.isna(last["ema_slow"]):
            return None
        return "UP" if last["ema_fast"] >= last["ema_slow"] else "DOWN"
    except Exception:
        return None


def scan_symbol(feed, settings, symbol: str, *, open_side=None,
                weights: Optional[Dict[str, float]] = None) -> SymbolScan:
    """Run the full strategy + scoring for one symbol. Never raises.

    ``open_side`` (PositionSide|None) is passed to the strategy so an already-open
    position yields HOLD/CLOSE rather than a fresh entry.
    """
    res = SymbolScan(symbol=symbol, available=True)
    try:
        if hasattr(feed, "ensure_symbol") and not feed.ensure_symbol(symbol):
            res.available = False
            res.skip_reason = "not offered by broker"
            return res

        candles = feed.get_candles(symbol=symbol, timeframe=settings.timeframe,
                                   n=settings.history_bars)
        if candles is None or len(candles) < settings.ema_slow + 2:
            res.available = False
            res.skip_reason = "insufficient history"
            return res

        enriched = add_indicators(candles, ema_fast=settings.ema_fast,
                                  ema_slow=settings.ema_slow,
                                  rsi_period=settings.rsi_period,
                                  atr_period=settings.atr_period)
        sig = evaluate(enriched, rsi_overbought=settings.rsi_overbought,
                       rsi_oversold=settings.rsi_oversold, open_side=open_side)

        res.action = sig.action.value
        res.reason = sig.reason
        res.price = float(sig.price) if sig.price == sig.price else 0.0
        res.rsi = sig.rsi
        res.atr = sig.atr

        # Live spread (best-effort; 0 if offline).
        spread = 0.0
        try:
            tick = feed.get_tick(symbol)
            spread = float(getattr(tick, "spread", 0.0) or 0.0)
        except Exception:
            spread = 0.0
        res.spread = spread

        if res.atr == res.atr and res.atr > 0 and res.price > 0:
            res.atr_pct = res.atr / res.price
            stop_distance = res.atr * settings.atr_multiplier
            res.spread_frac = (spread / stop_distance) if stop_distance > 0 else 0.0

        res.htf_trend = _htf_trend(feed, settings, symbol)

        sc = score_opportunity(
            action=sig.action, ema_fast=sig.ema_fast, ema_slow=sig.ema_slow,
            atr=sig.atr, price=sig.price, rsi=sig.rsi, spread=spread,
            atr_multiplier=settings.atr_multiplier, htf_trend=res.htf_trend,
            weights=weights,
        )
        res.score = sc.total
        res.tradable = sc.tradable
        res.skip_reason = sc.skip_reason
        res.components = {"signal": sc.signal, "volatility": sc.volatility,
                          "trend": sc.trend, "spread": sc.spread}
        return res
    except Exception as e:  # never let one bad symbol kill the scan
        res.error = str(e)
        res.skip_reason = f"error: {e}"
        res.tradable = False
        return res


def rank_opportunities(scans: List[SymbolScan]) -> List[SymbolScan]:
    """Tradable BUY/SELL opportunities, highest score first."""
    actionable = [s for s in scans if s.tradable and s.action in ("BUY", "SELL")]
    return sorted(actionable, key=lambda s: s.score, reverse=True)


def scan_watchlist(feed, settings, *, open_symbols=None,
                   weights: Optional[Dict[str, float]] = None) -> List[SymbolScan]:
    """Scan every symbol in ``settings.symbols``. ``open_symbols`` maps symbol->side
    for positions already open (so those evaluate for HOLD/CLOSE, not new entries)."""
    open_symbols = open_symbols or {}
    out: List[SymbolScan] = []
    for sym in settings.symbols:
        out.append(scan_symbol(feed, settings, sym,
                               open_side=open_symbols.get(sym), weights=weights))
    return out
