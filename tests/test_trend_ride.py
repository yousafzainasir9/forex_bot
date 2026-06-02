"""Unit tests for the trend-ride exit (let winners run after the target).

Behaviour under test (reversal detected on candle EXTREMES — higher highs for a
long, lower lows for a short):
  * Before the target is touched: never close, never armed.
  * The bar that FIRST touches the target: only flag it (never close that bar).
  * Once armed: a LONG closes when a bar's high is below the prior bar's high
    (the up-move stalled); a SHORT closes when a bar's low is above the prior
    bar's low (the down-move stalled).
"""

from bot.strategy import PositionSide, trend_ride_exit


def ride(**kw):
    base = dict(side=PositionSide.LONG, take_profit=1.32,
                prev_high=1.30, prev_low=1.29,
                last_high=1.30, last_low=1.29, tp_reached=False)
    base.update(kw)
    return trend_ride_exit(**base)


# --- before the target is reached: never close, never armed ---
def test_long_before_tp_holds():
    d = ride(last_high=1.31, last_low=1.29, prev_high=1.315)  # high < tp, lower high
    assert d.should_close is False and d.tp_reached is False


# --- the bar that first hits the target only flags (the 1.32 -> 1.33 step) ---
def test_long_target_bar_flags_only():
    # high reaches 1.33 (>= tp 1.32) AND it's a lower high than prior — still hold.
    d = ride(last_high=1.33, prev_high=1.40)
    assert d.tp_reached is True and d.should_close is False


# --- armed: keeps riding while it makes higher highs (1.44 after 1.33) ---
def test_long_armed_holds_on_higher_high():
    d = ride(last_high=1.44, prev_high=1.40, tp_reached=True)
    assert d.should_close is False


def test_long_armed_holds_on_equal_high():
    d = ride(last_high=1.44, prev_high=1.44, tp_reached=True)
    assert d.should_close is False


# --- armed: closes when a bar's high comes in lower (1.43 after 1.44) ---
def test_long_armed_exits_on_lower_high():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True)
    assert d.should_close is True


# --- SHORT mirror ---
def test_short_target_bar_flags_only():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.95, prev_high=0.97,
                        last_low=0.99, last_high=1.01, tp_reached=False)
    assert d.tp_reached is True and d.should_close is False


def test_short_armed_holds_on_lower_low():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.95, prev_high=0.97,
                        last_low=0.93, last_high=0.96, tp_reached=True)
    assert d.should_close is False


def test_short_armed_exits_on_higher_low():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.93, prev_high=0.96,
                        last_low=0.94, last_high=0.97, tp_reached=True)
    assert d.should_close is True


def test_disabled_when_no_target():
    d = ride(take_profit=0.0, last_high=2.0, prev_high=1.0)
    assert d.should_close is False and d.tp_reached is False
