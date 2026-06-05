"""Unit tests for the trend-ride exit (let winners run after the target).

Updated behaviour:
  * Before the target is touched: never close, never armed.
  * The bar that FIRST touches the target: only flag it (never close that bar).
  * Once armed, the move is only "over" after TWO consecutive stall bars
    (a long stalls on a lower high; a short on a higher low). A single stall
    bar holds. The streak resets when the move resumes.
  * The trend-ride exit fires only when closing would be IN PROFIT (close beyond
    entry). If entry/last_close are not supplied, the profit check is skipped.
"""

from bot.strategy import PositionSide, trend_ride_exit


def ride(**kw):
    base = dict(side=PositionSide.LONG, take_profit=1.32,
                prev_high=1.30, prev_low=1.29,
                last_high=1.30, last_low=1.29, tp_reached=False)
    base.update(kw)
    return trend_ride_exit(**base)


# --- before the target: never close, never armed ---
def test_long_before_tp_holds():
    d = ride(last_high=1.31, last_low=1.29, prev_high=1.315)
    assert d.should_close is False and d.tp_reached is False


def test_long_target_bar_flags_only():
    d = ride(last_high=1.33, prev_high=1.40)
    assert d.tp_reached is True and d.should_close is False


def test_long_armed_holds_on_higher_high():
    d = ride(last_high=1.44, prev_high=1.40, tp_reached=True)
    assert d.should_close is False


def test_long_armed_holds_on_equal_high():
    d = ride(last_high=1.44, prev_high=1.44, tp_reached=True)
    assert d.should_close is False


# --- one stall bar no longer closes (needs two) ---
def test_long_one_lower_high_holds():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True)
    assert d.should_close is False and d.stall_streak == 1


# --- two consecutive lower highs closes (profit check skipped here) ---
def test_long_two_lower_highs_exits():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True, stall_streak=1)
    assert d.should_close is True and d.stall_streak == 2


# --- a fresh higher high resets the stall streak ---
def test_long_higher_high_resets_streak():
    d = ride(last_high=1.45, prev_high=1.44, tp_reached=True, stall_streak=1)
    assert d.should_close is False and d.stall_streak == 0


# --- profit guard: two stalls but NOT in profit -> keep holding ---
def test_long_two_stalls_not_in_profit_holds():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True, stall_streak=1,
             entry=1.50, last_close=1.45)
    assert d.should_close is False


# --- profit guard: two stalls AND in profit -> exit ---
def test_long_two_stalls_in_profit_exits():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True, stall_streak=1,
             entry=1.40, last_close=1.45)
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


def test_short_one_higher_low_holds():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.93, prev_high=0.96,
                        last_low=0.94, last_high=0.97, tp_reached=True)
    assert d.should_close is False and d.stall_streak == 1


def test_short_two_higher_lows_exits():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.93, prev_high=0.96,
                        last_low=0.94, last_high=0.97, tp_reached=True,
                        stall_streak=1)
    assert d.should_close is True and d.stall_streak == 2


def test_short_two_stalls_not_in_profit_holds():
    d = trend_ride_exit(side=PositionSide.SHORT, take_profit=1.00,
                        prev_low=0.93, prev_high=0.96,
                        last_low=0.94, last_high=0.97, tp_reached=True,
                        stall_streak=1, entry=0.90, last_close=0.95)
    assert d.should_close is False


def test_disabled_when_no_target():
    d = ride(take_profit=0.0, last_high=2.0, prev_high=1.0)
    assert d.should_close is False and d.tp_reached is False


# --- ATR noise band: a minimal lower-high within noise_tol is NOT a stall ---
def test_long_lower_high_within_noise_band_holds():
    d = ride(last_high=1.439, prev_high=1.44, tp_reached=True, stall_streak=1,
             noise_tol=0.005)
    assert d.should_close is False and d.stall_streak == 0


def test_long_lower_high_beyond_noise_band_counts():
    d = ride(last_high=1.43, prev_high=1.44, tp_reached=True, stall_streak=1,
             noise_tol=0.005)
    assert d.should_close is True
