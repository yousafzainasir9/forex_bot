"""Tests for the pure MT5-deal -> ClosedTrade reconciliation builder."""

from dataclasses import dataclass

from bot.execution import build_closed_trades_from_deals


@dataclass
class FakeDeal:
    magic: int
    position_id: int
    entry: int          # 0=IN, 1=OUT
    type: int           # 0=BUY, 1=SELL
    time: float
    price: float
    volume: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0
    symbol: str = "EURUSD"


MAGIC = 250531
T_IN = 1_748_700_000   # arbitrary epoch
T_OUT = 1_748_703_600


def test_winning_long_trip_recorded_with_net_pnl():
    deals = [
        FakeDeal(MAGIC, 111, entry=0, type=0, time=T_IN, price=1.1000, volume=0.5,
                 profit=0.0, commission=-3.0),
        FakeDeal(MAGIC, 111, entry=1, type=1, time=T_OUT, price=1.1030, volume=0.5,
                 profit=150.0, commission=-3.0, swap=-1.0),
    ]
    open_map = {"111": {"risk_amount": 100.0, "stop_loss": 1.0985,
                        "take_profit": 1.10225, "reason_open": "test long"}}
    trades = build_closed_trades_from_deals(deals, MAGIC, set(), open_map)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "LONG"
    # net = 150 - 3 - 3 - 1 = 143
    assert t.pnl == 143.0
    assert t.commission == -6.0
    assert t.swap == -1.0
    # R = net / risk = 143 / 100
    assert t.r_multiple == 1.43
    assert t.position_id == 111
    assert t.entry == 1.1000 and t.exit == 1.1030


def test_losing_short_trip():
    deals = [
        FakeDeal(MAGIC, 222, entry=0, type=1, time=T_IN, price=1.2000, volume=0.2, profit=0.0),
        FakeDeal(MAGIC, 222, entry=1, type=0, time=T_OUT, price=1.2030, volume=0.2, profit=-60.0),
    ]
    trades = build_closed_trades_from_deals(deals, MAGIC, set(), {})
    assert len(trades) == 1
    assert trades[0].side == "SHORT"
    assert trades[0].pnl == -60.0
    assert trades[0].r_multiple == 0.0  # no open_map → unknown risk


def test_other_magic_ignored():
    deals = [
        FakeDeal(999, 333, entry=0, type=0, time=T_IN, price=1.1, volume=0.1, profit=0.0),
        FakeDeal(999, 333, entry=1, type=1, time=T_OUT, price=1.1, volume=0.1, profit=10.0),
    ]
    assert build_closed_trades_from_deals(deals, MAGIC, set(), {}) == []


def test_already_recorded_position_skipped():
    deals = [
        FakeDeal(MAGIC, 444, entry=0, type=0, time=T_IN, price=1.1, volume=0.1, profit=0.0),
        FakeDeal(MAGIC, 444, entry=1, type=1, time=T_OUT, price=1.1, volume=0.1, profit=20.0),
    ]
    assert build_closed_trades_from_deals(deals, MAGIC, {444}, {}) == []


def test_still_open_position_not_recorded():
    # Only an IN deal, no OUT yet → not a completed trade.
    deals = [
        FakeDeal(MAGIC, 555, entry=0, type=0, time=T_IN, price=1.1, volume=0.1, profit=0.0),
    ]
    assert build_closed_trades_from_deals(deals, MAGIC, set(), {}) == []
