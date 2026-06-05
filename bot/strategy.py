"""Module 4 — Strategy / Signal Engine.

Turns an indicator-enriched candle DataFrame into a single, explicit trading
``Signal`` for the most recently *closed* bar, with a human-readable reason.

Rule set (spec §4) — a transparent learning baseline, NOT a money-maker:
  * LONG  when fast EMA crosses ABOVE slow EMA AND RSI confirms momentum
          (confirm mode: RSI >= midline; legacy filter mode: RSI < overbought).
  * SHORT when fast EMA crosses BELOW slow EMA AND RSI confirms momentum
          (confirm mode: RSI <= midline; legacy filter mode: RSI > oversold).
  * EXIT  an open trade early on the opposite EMA cross.
  * Otherwise HOLD. Sitting out is a valid action.

A "cross" is evaluated on the two most recent closed bars:
  fast[t-1] <= slow[t-1]  and  fast[t] > slow[t]   → bullish cross
  fast[t-1] >= slow[t-1]  and  fast[t] < slow[t]   → bearish cross

Because we operate only on closed bars, there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # exit an existing position (opposite-cross exit)


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Signal:
    action: Action
    reason: str
    price: float          # close of the signal bar (reference entry price)
    atr: float            # ATR at the signal bar (for stop sizing downstream)
    rsi: float
    ema_fast: float
    ema_slow: float
    bar_time: Optional[pd.Timestamp] = None

    def __str__(self) -> str:
        t = self.bar_time.isoformat() if self.bar_time is not None else "?"
        return f"{t} {self.action.value}: {self.reason}"


def _bullish_cross(prev_fast, prev_slow, fast, slow) -> bool:
    return prev_fast <= prev_slow and fast > slow


def _bearish_cross(prev_fast, prev_slow, fast, slow) -> bool:
    return prev_fast >= prev_slow and fast < slow


def evaluate(
    df: pd.DataFrame,
    *,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    rsi_mode: str = "filter",
    rsi_midline: float = 50.0,
    adx_min: float = 0.0,
    open_side: Optional[PositionSide] = None,
) -> Signal:
    """Evaluate the strategy on the last closed bar of ``df``.

    Parameters
    ----------
    df : indicator-enriched DataFrame (must contain ema_fast, ema_slow, rsi, atr).
         The LAST row is treated as the most recently *closed* bar.
    rsi_overbought / rsi_oversold : RSI thresholds used in legacy "filter" mode.
    rsi_mode : how RSI gates an entry.
        * "confirm" (recommended for this trend-following cross): RSI must AGREE
          with the cross — a BUY needs RSI >= ``rsi_midline``, a SELL needs
          RSI <= ``rsi_midline``. This requires momentum to back the signal.
        * "filter" (legacy): only veto when already overbought/oversold — which,
          at a fresh cross, almost never triggers (near no-op).
    rsi_midline : the confirm-mode threshold (default 50).
    adx_min : minimum ADX (trend strength) required to OPEN a trade; <= 0 disables
         the gate. Requires an "adx" column when > 0. Exits are never gated.
    open_side : the side of a currently-open position (or None). Used to decide
         whether an opposite cross should produce a CLOSE.

    Returns
    -------
    Signal for the last bar. Never raises on normal data; returns HOLD if there
    is not enough history or indicators are still warming up (NaN).
    """
    needed = {"ema_fast", "ema_slow", "rsi", "atr", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Strategy input missing columns: {sorted(missing)}")

    if len(df) < 2:
        return _hold(df, "not enough bars to evaluate a cross")

    cur = df.iloc[-1]
    prev = df.iloc[-2]
    bar_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None

    # Indicators still warming up → do nothing.
    warmup_cols = [cur["ema_fast"], cur["ema_slow"], cur["rsi"], cur["atr"],
                   prev["ema_fast"], prev["ema_slow"]]
    if any(pd.isna(v) for v in warmup_cols):
        return _hold(df, "indicators warming up (NaN)")

    fast, slow, r, a = cur["ema_fast"], cur["ema_slow"], cur["rsi"], cur["atr"]
    pf, ps = prev["ema_fast"], prev["ema_slow"]

    bull = _bullish_cross(pf, ps, fast, slow)
    bear = _bearish_cross(pf, ps, fast, slow)

    # --- Exit logic first: an open position closes on the opposite cross. ---
    if open_side is PositionSide.LONG and bear:
        return _signal(Action.CLOSE, cur, bar_time,
                       "bearish EMA cross — exit existing LONG")
    if open_side is PositionSide.SHORT and bull:
        return _signal(Action.CLOSE, cur, bar_time,
                       "bullish EMA cross — exit existing SHORT")

    # If a position is already open and there's no opposite cross, hold it.
    if open_side is not None:
        return _signal(Action.HOLD, cur, bar_time,
                       f"holding open {open_side.value}; no opposite cross")

    # --- Trend-strength gate (skip chop). Applies only to fresh entries; an open
    #     position's exits were already handled above. adx_min<=0 disables it. ---
    if adx_min > 0 and (bull or bear):
        adx_val = (float(cur["adx"])
                   if "adx" in cur.index and not pd.isna(cur["adx"]) else float("nan"))
        if pd.isna(adx_val) or adx_val < adx_min:
            return _signal(Action.HOLD, cur, bar_time,
                           f"cross but ADX {adx_val:.1f} < {adx_min:.0f} (no trend strength)")

    # --- Entry logic (flat). ---
    if bull:
        if rsi_mode == "confirm":
            if r >= rsi_midline:
                return _signal(Action.BUY, cur, bar_time,
                               f"bullish EMA cross & RSI {r:.1f} >= {rsi_midline:.0f} (momentum confirms)")
            return _signal(Action.HOLD, cur, bar_time,
                           f"bullish cross but RSI {r:.1f} < {rsi_midline:.0f} (no momentum confirmation)")
        # legacy filter mode
        if r < rsi_overbought:
            return _signal(Action.BUY, cur, bar_time,
                           f"bullish EMA cross & RSI {r:.1f} < {rsi_overbought:.0f}")
        return _signal(Action.HOLD, cur, bar_time,
                       f"bullish cross but RSI {r:.1f} >= {rsi_overbought:.0f} (overbought)")

    if bear:
        if rsi_mode == "confirm":
            if r <= rsi_midline:
                return _signal(Action.SELL, cur, bar_time,
                               f"bearish EMA cross & RSI {r:.1f} <= {rsi_midline:.0f} (momentum confirms)")
            return _signal(Action.HOLD, cur, bar_time,
                           f"bearish cross but RSI {r:.1f} > {rsi_midline:.0f} (no momentum confirmation)")
        # legacy filter mode
        if r > rsi_oversold:
            return _signal(Action.SELL, cur, bar_time,
                           f"bearish EMA cross & RSI {r:.1f} > {rsi_oversold:.0f}")
        return _signal(Action.HOLD, cur, bar_time,
                       f"bearish cross but RSI {r:.1f} <= {rsi_oversold:.0f} (oversold)")

    return _signal(Action.HOLD, cur, bar_time, "no EMA cross")


@dataclass(frozen=True)
class TrendRideDecision:
    """Result of the trend-ride exit check once a trade reaches its target."""
    should_close: bool
    tp_reached: bool
    reason: str
    stall_streak: int = 0


def trend_ride_exit(
    *,
    side: PositionSide,
    take_profit: float,
    prev_high: float,
    prev_low: float,
    last_high: float,
    last_low: float,
    tp_reached: bool,
    stall_streak: int = 0,
    bars_to_exit: int = 2,
    entry: Optional[float] = None,
    last_close: Optional[float] = None,
    noise_tol: float = 0.0,
) -> TrendRideDecision:
    """Trend-ride exit: once price reaches the take-profit ("upper limit") it does
    NOT close. Instead it rides the move and exits only when the trend turns,
    detected on the candle EXTREMES:

      * LONG  — ride while each new bar makes a HIGHER (or equal) high; close the
                moment a bar's high comes in BELOW the prior bar's high. Example:
                target 1.32 hit at 1.33 → flag; price runs to 1.44; next bar's
                high is 1.43 (< 1.44) → the up-move stalled → close.
      * SHORT — mirror: ride while each new bar makes a LOWER (or equal) low;
                close when a bar's low comes in ABOVE the prior bar's low.

    The hard stop-loss (the "lower limit" for a long) is enforced separately by
    the broker and always closes the trade immediately.

    The TP-reached state latches: the bar that first touches the target only
    flags it (never closes on that bar); subsequent bars decide the exit. Pure and
    deterministic so it can be unit-tested and reused by the backtest.

    Parameters
    ----------
    side             : LONG or SHORT.
    take_profit      : the target ("upper limit" for a long); <= 0 disables it.
    prev_high/low    : high/low of the bar before the last closed bar.
    last_high/low    : high/low of the most recently closed bar.
    tp_reached       : whether the target was already reached on a previous bar.

    Returns
    -------
    TrendRideDecision(should_close, tp_reached, reason).
    """
    reached = bool(tp_reached)
    newly_reached = False
    if not reached and take_profit and take_profit > 0:
        if side is PositionSide.LONG and last_high >= take_profit:
            newly_reached = True
        elif side is PositionSide.SHORT and last_low <= take_profit:
            newly_reached = True
    reached = reached or newly_reached

    if not reached:
        return TrendRideDecision(False, reached, "target not reached — holding for TP")

    # The bar that FIRST reaches the target only flags it — never closes on this
    # bar, even if it turns down. We wait for a subsequent bar to confirm a turn.
    if newly_reached:
        return TrendRideDecision(False, reached,
                                 "target reached this bar — flagged, holding to next bar")

    # Already armed on an earlier bar. The move is only declared "over" after
    # ``bars_to_exit`` CONSECUTIVE stall bars (a single stall no longer closes the
    # trade), and the exit only fires if it would bank a PROFIT. Otherwise we keep
    # riding and let the broker stop-loss / break-even stop cap the downside.
    # ``noise_tol`` (ATR-relative) ignores a minimal lower-high / higher-low as
    # market noise so tiny red candles / dojis don't end the ride.
    tol = max(0.0, float(noise_tol))
    if side is PositionSide.LONG:
        stalled = last_high < prev_high - tol
        riding_reason = "riding LONG trend - still making higher highs"
    else:
        stalled = last_low > prev_low + tol
        riding_reason = "riding SHORT trend - still making lower lows"

    if not stalled:
        # Move resumed -> reset the stall streak.
        return TrendRideDecision(False, reached, riding_reason, 0)

    streak = int(stall_streak) + 1
    if streak < bars_to_exit:
        return TrendRideDecision(
            False, reached,
            f"trend-ride: stall bar {streak}/{bars_to_exit} - holding for confirmation",
            streak)

    # Enough consecutive stall bars: only exit if doing so banks a profit.
    if entry is not None and last_close is not None:
        in_profit = (last_close > entry if side is PositionSide.LONG
                     else last_close < entry)
        if not in_profit:
            return TrendRideDecision(
                False, reached,
                f"trend-ride: {streak} stall bars but not in profit "
                f"(close {last_close:.5f} vs entry {entry:.5f}) - holding; stop protects",
                streak)

    extreme = "high" if side is PositionSide.LONG else "low"
    return TrendRideDecision(
        True, reached,
        f"trend-ride exit: {streak} consecutive stall {extreme}-bars, in profit - move over",
        streak)


def _signal(action: Action, row: pd.Series, bar_time, reason: str) -> Signal:
    return Signal(
        action=action,
        reason=reason,
        price=float(row["close"]),
        atr=float(row["atr"]) if not pd.isna(row["atr"]) else float("nan"),
        rsi=float(row["rsi"]) if not pd.isna(row["rsi"]) else float("nan"),
        ema_fast=float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else float("nan"),
        ema_slow=float(row["ema_slow"]) if not pd.isna(row["ema_slow"]) else float("nan"),
        bar_time=bar_time,
    )


def _hold(df: pd.DataFrame, reason: str) -> Signal:
    if len(df) == 0:
        return Signal(Action.HOLD, reason, float("nan"), float("nan"),
                      float("nan"), float("nan"), float("nan"), None)
    row = df.iloc[-1]
    bar_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    return _signal(Action.HOLD, row, bar_time, reason)
