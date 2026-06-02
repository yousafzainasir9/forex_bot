"""Unit tests for the risk manager — sizing, stops/targets, and veto paths."""

import math

import pandas as pd
import pytest

from bot.risk import Decision, DailyState, RiskManager, SymbolSpec
from bot.strategy import Action, PositionSide, Signal


def _signal(action=Action.BUY, price=1.1000, atr=0.0010, rsi=50.0):
    return Signal(
        action=action, reason="test", price=price, atr=atr, rsi=rsi,
        ema_fast=1.1, ema_slow=1.0, bar_time=pd.Timestamp("2024-01-01", tz="UTC"),
    )


def _rm(**kw):
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    defaults = dict(risk_per_trade=0.01, max_daily_loss=0.03, max_open_positions=1,
                    atr_multiplier=1.5, risk_reward=1.5, symbol_spec=spec)
    defaults.update(kw)
    return RiskManager(**defaults)


def test_long_sizing_and_levels():
    rm = _rm()
    d = rm.evaluate(_signal(Action.BUY, price=1.1000, atr=0.0010),
                    equity=10_000, open_positions=0, daily=DailyState(10_000))
    assert d.decision is Decision.APPROVE
    p = d.plan
    assert p.stop_distance == pytest.approx(0.0015, abs=1e-9)
    assert p.side is PositionSide.LONG
    assert p.stop_loss == pytest.approx(1.0985, abs=1e-6)
    assert p.take_profit == pytest.approx(1.10225, abs=1e-6)
    expected_lots = math.floor((100 / 150) / 0.01 + 1e-9) * 0.01
    assert p.lots == pytest.approx(round(expected_lots, 2))
    assert p.risk_amount <= 100 * 1.01


def test_short_levels_inverted():
    rm = _rm()
    d = rm.evaluate(_signal(Action.SELL, price=1.2000, atr=0.0020),
                    equity=10_000, open_positions=0, daily=DailyState(10_000))
    p = d.plan
    assert p.side is PositionSide.SHORT
    assert p.stop_loss == pytest.approx(1.2030, abs=1e-6)
    assert p.take_profit == pytest.approx(1.1955, abs=1e-6)


def test_kill_switch_vetoes():
    rm = _rm()
    d = rm.evaluate(_signal(), equity=10_000, open_positions=0,
                    daily=DailyState(10_000), kill_switch=True)
    assert d.decision is Decision.VETO
    assert "kill switch" in d.reason.lower()


def test_max_positions_vetoes():
    rm = _rm(max_open_positions=1)
    d = rm.evaluate(_signal(), equity=10_000, open_positions=1, daily=DailyState(10_000))
    assert d.decision is Decision.VETO
    assert "max open" in d.reason.lower()


def test_daily_loss_halt_vetoes():
    rm = _rm(max_daily_loss=0.03)
    daily = DailyState(start_equity=10_000, realized_pnl_today=-310)
    d = rm.evaluate(_signal(), equity=9_690, open_positions=0, daily=daily)
    assert d.decision is Decision.VETO
    assert "halted" in d.reason.lower()


def test_zero_atr_vetoes():
    rm = _rm()
    d = rm.evaluate(_signal(atr=0.0), equity=10_000, open_positions=0, daily=DailyState(10_000))
    assert d.decision is Decision.VETO
    assert "atr" in d.reason.lower()


def test_live_without_allow_vetoes():
    rm = _rm()
    d = rm.evaluate(_signal(), equity=10_000, open_positions=0, daily=DailyState(10_000),
                    is_live=True, allow_live=False)
    assert d.decision is Decision.VETO


def test_hold_and_close_not_sized():
    rm = _rm()
    for action in (Action.HOLD, Action.CLOSE):
        d = rm.evaluate(_signal(action=action), equity=10_000, open_positions=0,
                        daily=DailyState(10_000))
        assert d.decision is Decision.VETO


def test_tight_stop_small_account_vetoes_overrisk():
    spec = SymbolSpec(contract_size=100_000, volume_min=0.10, volume_max=100,
                      volume_step=0.10, digits=5, point=0.00001)
    rm = _rm(symbol_spec=spec)
    d = rm.evaluate(_signal(atr=0.0010), equity=200, open_positions=0,
                    daily=DailyState(200))
    assert d.decision is Decision.VETO


def test_loss_fraction_only_counts_losses():
    assert DailyState(10_000, realized_pnl_today=500).loss_fraction() == 0.0
    assert DailyState(10_000, realized_pnl_today=-200).loss_fraction() == pytest.approx(0.02)


def test_small_account_trades_tight_stop_and_reports_real_pct():
    # $50 account, 0.01 min lot, tight ~3-pip stop -> should APPROVE at min lot.
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    rm = _rm(symbol_spec=spec, max_risk_per_trade=0.02)
    d = rm.evaluate(_signal(atr=0.0002), equity=50.0, open_positions=0,
                    daily=DailyState(50.0))
    assert d.decision is Decision.APPROVE
    assert d.plan.lots == 0.01
    assert "of equity" in d.reason


def test_small_account_wide_stop_vetoes_over_ceiling():
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    rm = _rm(symbol_spec=spec, max_risk_per_trade=0.02)
    d = rm.evaluate(_signal(atr=0.0008), equity=50.0, open_positions=0,
                    daily=DailyState(50.0))
    assert d.decision is Decision.VETO
    assert "ceiling" in d.reason


def test_raising_ceiling_allows_wider_stop():
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    rm = _rm(symbol_spec=spec, max_risk_per_trade=0.05)  # allow up to 5%
    d = rm.evaluate(_signal(atr=0.0008), equity=50.0, open_positions=0,
                    daily=DailyState(50.0))
    assert d.decision is Decision.APPROVE


def test_tier_for_equity_boundaries():
    from bot.risk import tier_for_equity
    assert tier_for_equity(50)["risk_per_trade"] == 0.20
    assert tier_for_equity(99)["risk_per_trade"] == 0.20
    assert tier_for_equity(100)["risk_per_trade"] == 0.10
    assert tier_for_equity(200)["risk_per_trade"] == 0.07
    assert tier_for_equity(300)["risk_per_trade"] == 0.05
    assert tier_for_equity(600)["risk_per_trade"] == 0.03
    assert tier_for_equity(1000)["risk_per_trade"] == 0.01
    assert tier_for_equity(50000)["risk_per_trade"] == 0.01


def test_tiered_mode_scales_risk_by_equity():
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    rm = RiskManager(risk_mode="tiered", atr_multiplier=1.5, risk_reward=1.5, symbol_spec=spec)
    # $50 -> ~20% risk
    d50 = rm.evaluate(_signal(atr=0.0002), equity=50.0, open_positions=0, daily=DailyState(50.0))
    assert d50.decision is Decision.APPROVE
    assert d50.plan.risk_amount == pytest.approx(50.0 * 0.20, rel=0.1)
    # $1000 -> ~1% risk
    d1k = rm.evaluate(_signal(atr=0.0002), equity=1000.0, open_positions=0, daily=DailyState(1000.0))
    assert d1k.plan.risk_amount == pytest.approx(1000.0 * 0.01, rel=0.15)


def test_fixed_mode_unaffected_by_tiers():
    spec = SymbolSpec(contract_size=100_000, volume_min=0.01, volume_max=100,
                      volume_step=0.01, digits=5, point=0.00001)
    rm = RiskManager(risk_mode="fixed", risk_per_trade=0.01, atr_multiplier=1.5,
                     risk_reward=1.5, symbol_spec=spec)
    d = rm.evaluate(_signal(atr=0.0002), equity=50.0, open_positions=0, daily=DailyState(50.0))
    # fixed 1% of $50 = $0.50 target; min lot 0.01 risks ~$0.30 -> approve at min lot
    assert d.decision is Decision.APPROVE
    assert d.plan.lots == 0.01
