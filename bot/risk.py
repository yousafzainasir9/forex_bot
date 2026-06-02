"""Module 5 — Risk Manager (non-negotiable).

The risk manager sits between the strategy and execution. It can VETO any trade
the strategy proposes. Risk rules always beat signals.

Responsibilities (spec §5):
  * Position sizing: risk a fixed fraction of equity (default 1%) given the
    ATR-based stop distance. Lot size is derived, never guessed.
  * Stop-loss: entry ± ATR × multiplier.
  * Take-profit: risk:reward of 1:R (default 1.5) from the stop distance.
  * Max open positions (default 1).
  * Max daily loss → halt trading for the rest of the day.
  * Kill switch / mode checks.

This module is pure: it consumes plain numbers / dataclasses and returns a
decision. It never touches MT5 directly, which keeps it fully unit-testable.

NOTE on lot/pip maths: forex position value depends on the instrument's contract
size and the value of one price unit per lot ("value per price unit per lot").
We pass these in as `SymbolSpec` so the manager has no broker dependency. For a
standard FX pair, contract_size is typically 100_000 and the stop loss in money
terms equals: lots * contract_size * stop_distance_in_price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .strategy import Action, PositionSide, Signal


class Decision(str, Enum):
    APPROVE = "APPROVE"
    VETO = "VETO"


# Account-size risk tiers (used when risk_mode="tiered"). Smaller balances trade
# more aggressively so the broker's 0.01 min lot still produces meaningful moves;
# from $1,000 up it reverts to the conservative 1% baseline. Tunable.
#   risk_per_trade     = fraction of equity risked per trade
#   max_risk_per_trade = ceiling: veto if the MIN lot would risk more than this
#   max_daily_loss     = halt trading for the day past this loss fraction
DEFAULT_RISK_TIERS = [
    {"min_equity": 1000.0, "risk_per_trade": 0.01, "max_risk_per_trade": 0.02, "max_daily_loss": 0.03},
    {"min_equity": 500.0,  "risk_per_trade": 0.03, "max_risk_per_trade": 0.05, "max_daily_loss": 0.08},
    {"min_equity": 250.0,  "risk_per_trade": 0.05, "max_risk_per_trade": 0.07, "max_daily_loss": 0.12},
    {"min_equity": 150.0,  "risk_per_trade": 0.07, "max_risk_per_trade": 0.10, "max_daily_loss": 0.18},
    {"min_equity": 100.0,  "risk_per_trade": 0.10, "max_risk_per_trade": 0.14, "max_daily_loss": 0.25},
    {"min_equity": 0.0,    "risk_per_trade": 0.20, "max_risk_per_trade": 0.25, "max_daily_loss": 0.40},
]


def breaker_reason(
    *,
    consecutive_losses: int,
    max_consecutive_losses: int,
    drawdown_fraction: float,
    max_total_drawdown: float,
) -> Optional[str]:
    """Return a halt reason if a global circuit-breaker has tripped, else None.

    Pure and testable. Each check is disabled when its limit is <= 0. These are
    account-level protections that sit ON TOP of the per-day loss halt:
      * a losing streak (``consecutive_losses``) — forces a review after a bad run;
      * a peak-to-current equity drawdown (``drawdown_fraction``) — a hard stop on
        new entries while the account is deep underwater.
    """
    if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
        return (f"{consecutive_losses} consecutive losses >= limit "
                f"{max_consecutive_losses} — new entries halted")
    if max_total_drawdown > 0 and drawdown_fraction >= max_total_drawdown:
        return (f"account drawdown {drawdown_fraction:.1%} >= limit "
                f"{max_total_drawdown:.1%} — new entries halted")
    return None


def tier_for_equity(equity: float, tiers=None) -> dict:
    """Return the risk tier whose min_equity is the highest one <= equity."""
    tiers = tiers or DEFAULT_RISK_TIERS
    for tier in sorted(tiers, key=lambda t: t["min_equity"], reverse=True):
        if equity >= tier["min_equity"]:
            return tier
    return tiers[-1]


@dataclass(frozen=True)
class SymbolSpec:
    """Broker/instrument constants needed to size a position.

    Sensible FX defaults are provided. In live use these come from MT5's
    ``symbol_info`` (see data.py / execution.py).
    """
    contract_size: float = 100_000.0   # units per 1.0 lot (standard FX)
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    digits: int = 5                     # price decimal places
    point: float = 0.00001              # smallest price increment


@dataclass(frozen=True)
class TradePlan:
    """A fully-specified, risk-approved order ready for execution."""
    side: PositionSide
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    risk_amount: float       # money at risk if stop is hit
    stop_distance: float     # price distance entry→stop
    reason: str


@dataclass(frozen=True)
class RiskDecision:
    decision: Decision
    reason: str
    plan: Optional[TradePlan] = None

    @property
    def approved(self) -> bool:
        return self.decision is Decision.APPROVE


@dataclass
class DailyState:
    """Tracks intraday loss to enforce the daily-loss halt.

    `start_equity` should be set to account equity at the start of the trading
    day; `realized_pnl_today` accumulates closed-trade P&L for the day.
    """
    start_equity: float
    realized_pnl_today: float = 0.0

    def loss_fraction(self) -> float:
        if self.start_equity <= 0:
            return 0.0
        # Only losses count toward the halt; gains don't unlock more room beyond 0.
        return max(0.0, -self.realized_pnl_today) / self.start_equity


def _round_to_step(volume: float, step: float) -> float:
    if step <= 0:
        return volume
    # Floor to the nearest step so we never over-risk.
    return math.floor(volume / step + 1e-9) * step


class RiskManager:
    def __init__(
        self,
        *,
        risk_per_trade: float = 0.01,
        max_risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.03,
        risk_mode: str = "fixed",   # "fixed" uses the values above; "tiered" scales by equity
        tiers=None,
        max_open_positions: int = 1,
        atr_multiplier: float = 1.5,
        risk_reward: float = 1.5,
        symbol_spec: Optional[SymbolSpec] = None,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.risk_mode = risk_mode
        self.tiers = tiers or DEFAULT_RISK_TIERS
        self.max_open_positions = max_open_positions
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.spec = symbol_spec or SymbolSpec()

    def effective_risk(self, equity: float) -> dict:
        """Risk params for this equity: a tier (tiered mode) or the fixed values."""
        if self.risk_mode == "tiered":
            return tier_for_equity(equity, self.tiers)
        return {"risk_per_trade": self.risk_per_trade,
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_daily_loss": self.max_daily_loss}

    def evaluate(
        self,
        signal: Signal,
        *,
        equity: float,
        open_positions: int,
        daily: DailyState,
        kill_switch: bool = False,
        is_live: bool = False,
        allow_live: bool = False,
    ) -> RiskDecision:
        """Approve or veto a strategy signal, producing a TradePlan if approved.

        Only BUY / SELL signals can be approved into a TradePlan. HOLD and CLOSE
        are passed through to the runner (CLOSE is handled by execution, not sized).
        """
        # --- Global guards (apply before anything else). ---
        eff = self.effective_risk(equity if equity and equity > 0 else 0.0)
        rpt, mrpt, mdl = eff["risk_per_trade"], eff["max_risk_per_trade"], eff["max_daily_loss"]
        if kill_switch:
            return RiskDecision(Decision.VETO, "kill switch is ON — no new orders")

        if is_live and not allow_live:
            return RiskDecision(
                Decision.VETO,
                "mode is LIVE but live trading not explicitly allowed — refusing",
            )

        if daily.loss_fraction() >= mdl:
            return RiskDecision(
                Decision.VETO,
                f"daily loss {daily.loss_fraction():.2%} >= limit {mdl:.2%} — halted for the day",
            )

        # CLOSE / HOLD are not sized here.
        if signal.action in (Action.HOLD, Action.CLOSE):
            return RiskDecision(Decision.VETO, f"{signal.action.value}: nothing to size")

        if open_positions >= self.max_open_positions:
            return RiskDecision(
                Decision.VETO,
                f"already at max open positions ({open_positions}/{self.max_open_positions})",
            )

        if equity <= 0:
            return RiskDecision(Decision.VETO, "non-positive equity")

        atr = signal.atr
        if atr is None or math.isnan(atr) or atr <= 0:
            return RiskDecision(Decision.VETO, "invalid/zero ATR — cannot size a stop")

        entry = signal.price
        stop_distance = atr * self.atr_multiplier
        if stop_distance <= 0:
            return RiskDecision(Decision.VETO, "non-positive stop distance")

        side = PositionSide.LONG if signal.action is Action.BUY else PositionSide.SHORT
        if side is PositionSide.LONG:
            stop_loss = entry - stop_distance
            take_profit = entry + stop_distance * self.risk_reward
        else:
            stop_loss = entry + stop_distance
            take_profit = entry - stop_distance * self.risk_reward

        # --- Position sizing from money-at-risk and stop distance. ---
        risk_amount = equity * rpt
        # Money lost per 1.0 lot if stop is hit = contract_size * stop_distance.
        money_per_lot = self.spec.contract_size * stop_distance
        if money_per_lot <= 0:
            return RiskDecision(Decision.VETO, "degenerate contract/stop maths")

        raw_lots = risk_amount / money_per_lot
        lots = _round_to_step(raw_lots, self.spec.volume_step)
        lots = max(self.spec.volume_min, min(lots, self.spec.volume_max))

        # If even the minimum lot risks more than allowed, veto rather than over-risk.
        actual_risk = lots * money_per_lot
        if lots <= 0:
            return RiskDecision(Decision.VETO, "computed lot size rounded to zero")
        risk_ceiling = equity * mrpt
        if raw_lots < self.spec.volume_min and actual_risk > risk_ceiling:
            return RiskDecision(
                Decision.VETO,
                f"min lot {self.spec.volume_min} would risk {actual_risk:.2f} "
                f"({actual_risk / equity:.2%} of equity) > ceiling {risk_ceiling:.2f} "
                f"({self.max_risk_per_trade:.1%}). Raise MAX_RISK_PER_TRADE to allow, "
                f"or use a tighter-stop signal.",
            )

        plan = TradePlan(
            side=side,
            entry_price=round(entry, self.spec.digits),
            stop_loss=round(stop_loss, self.spec.digits),
            take_profit=round(take_profit, self.spec.digits),
            lots=round(lots, 2),
            risk_amount=round(actual_risk, 2),
            stop_distance=round(stop_distance, self.spec.digits),
            reason=(
                f"{side.value} {round(lots, 2)} lots @ {round(entry, self.spec.digits)}, "
                f"SL {round(stop_loss, self.spec.digits)} TP {round(take_profit, self.spec.digits)}, "
                f"risk ≈ {round(actual_risk, 2)} ({actual_risk / equity:.2%} of equity)"
            ),
        )
        return RiskDecision(Decision.APPROVE, plan.reason, plan)
