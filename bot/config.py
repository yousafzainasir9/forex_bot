"""Module 1 — Config & Control.

Single source of truth for every tunable parameter, credentials loading,
the DEMO/LIVE mode flag (defaults to DEMO), and the kill switch.

Design rules:
  * DEMO is the default. LIVE requires TWO explicit env values to line up, so it
    can never be entered by accident.
  * The kill switch, when on, makes the execution layer refuse all new orders.
  * Nothing here talks to MT5; this module is pure and import-safe everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a hard dep, but keep import-safe for partial installs.
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# The exact phrase a user must put in I_UNDERSTAND_LIVE_TRADING_RISKS to permit LIVE.
_LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_LIVE_TRADING_RISKS"

# Default multi-asset watchlist for the opportunity scanner. The bot probes each
# at startup and silently drops any your broker doesn't offer (names vary by
# broker, e.g. some use "GOLD" instead of "XAUUSD" or add a ".raw"/"m" suffix).
DEFAULT_WATCHLIST = [
    # FX majors (tightest spreads, most reliable on demo)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # FX crosses (more volatility, wider spreads)
    "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD",
    # Metals + oil (big intraday ranges)
    "XAUUSD", "XAGUSD", "USOIL",
    # Indices + crypto (high volatility; availability/spreads vary a lot)
    "US30", "US500", "NAS100", "GER40", "BTCUSD", "ETHUSD",
]


def _parse_symbols(raw: Optional[str], fallback: str) -> List[str]:
    """Parse a comma/space/semicolon-separated SYMBOLS list, upper-cased & deduped.

    Empty/unset -> the full DEFAULT_WATCHLIST (multi-asset by default). The literal
    value "SINGLE" -> just the primary SYMBOL (legacy single-symbol behaviour).
    """
    if raw is None or raw.strip() == "":
        return list(DEFAULT_WATCHLIST)
    if raw.strip().upper() == "SINGLE":
        return [fallback]
    parts = [p.strip().upper()
             for chunk in raw.split(",")
             for p in chunk.replace(";", " ").split()]
    out: List[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out or [fallback]


def in_trading_session(hour: int, start_hour: int, end_hour: int) -> bool:
    """True if ``hour`` (0-23 UTC) is inside the [start, end) trading window.

    Pure and testable. Supports windows that wrap past midnight (start > end);
    start == end is treated as a 24-hour, always-open window.
    """
    hour %= 24
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour  # wraps midnight


class TradingMode(str, Enum):
    DEMO = "DEMO"
    LIVE = "LIVE"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of all runtime settings. Build via `Settings.load()`."""

    # --- MT5 credentials ---
    mt5_login: Optional[int]
    mt5_password: Optional[str]
    mt5_server: Optional[str]
    mt5_terminal_path: Optional[str]

    # --- Mode / safety ---
    mode: TradingMode
    live_confirmed: bool
    kill_switch: bool

    # --- Instrument / timeframe ---
    symbol: str            # primary symbol (charted by default; legacy single-symbol)
    symbols: List[str]     # full watchlist the scanner ranks each loop
    timeframe: str         # e.g. "M5"; mapped to an MT5 constant in data.py

    # --- Multi-symbol scanner ---
    scan_mode: str         # "best" (one trade at a time), "top_n", or "any"
    scan_top_n: int        # max concurrent positions when scan_mode != "best"
    scan_min_score: float  # skip opportunities scoring below this (0..1); 0 = trade best
    htf_timeframe: str     # higher timeframe used for trend-alignment ranking (e.g. M15)
    require_htf_align: bool # HARD gate: only trade when the M5 signal agrees with the HTF trend

    # --- Strategy params ---
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    rsi_mode: str          # "confirm" (RSI must agree with the cross) or "filter" (legacy veto)
    rsi_midline: float     # confirm-mode threshold: long needs RSI >= this, short <= this
    atr_period: int
    adx_period: int        # ADX lookback (trend-strength)
    adx_min: float         # minimum ADX to allow an entry; ignored unless require_adx
    require_adx: bool      # HARD gate: only trade when ADX >= adx_min (skip chop)
    atr_multiplier: float
    risk_reward: float
    ride_trend_after_tp: bool  # once target hit, hold & trail on closes instead of closing at TP

    # --- Session filter (UTC) ---
    session_filter: bool   # when True, only OPEN new trades inside the hour window below
    session_start_hour: int  # inclusive UTC hour the trading window opens (0-23)
    session_end_hour: int    # exclusive UTC hour the window closes (1-24; may wrap past midnight)

    # --- Risk params ---
    risk_per_trade: float       # fraction of equity, e.g. 0.01 = 1%
    max_risk_per_trade: float   # hard ceiling: veto a trade whose min-lot risk exceeds this
    risk_mode: str              # 'tiered' (scale risk by equity) or 'fixed'
    max_daily_loss: float       # fraction of start-of-day equity, e.g. 0.03 = 3%
    max_open_positions: int
    max_total_drawdown: float   # halt new entries if peak->equity drawdown exceeds this (0 = off)
    max_consecutive_losses: int # halt new entries for the day after this many losses in a row (0 = off)
    max_per_currency: int       # cap concurrent positions sharing a currency (correlation guard)
    max_spread_points: int      # execution refuses a fill if live spread exceeds this (points; 0 = off)

    # --- Exit management ---
    trail_after_tp: bool        # once target hit, trail the stop by ATR (on top of break-even)
    trail_atr_mult: float       # trailing-stop distance = ATR x this
    partial_tp_enabled: bool    # bank part of the position at partial_tp_r, ride the rest
    partial_tp_fraction: float  # fraction of the position to close at the partial target (0..1)
    partial_tp_r: float         # partial target distance, in R (multiples of the stop distance)
    lock_profit_r: float        # once target hit, move stop to LOCK this many R of profit (0 = break-even)
    max_bars_in_trade: int      # force-close a trade that NEVER reached target after this many bars (0 = off)

    # --- Runtime ---
    history_bars: int           # how many candles to pull each cycle
    magic_number: int           # tags this bot's orders in MT5
    deviation_points: int       # max price slippage allowed on market orders
    broker_utc_offset_hours: Optional[float]  # None = auto-detect server->UTC offset
    reconcile_lookback_days: int  # how far back the startup backfill scans for closed trades
    log_dir: Path

    @property
    def is_live(self) -> bool:
        """True only when mode==LIVE AND the user explicitly confirmed the risks."""
        return self.mode is TradingMode.LIVE and self.live_confirmed

    @property
    def live_misconfigured(self) -> bool:
        """LIVE was requested but not properly confirmed — a hard error state."""
        return self.mode is TradingMode.LIVE and not self.live_confirmed

    @classmethod
    def load(cls, env_path: Optional[str | Path] = None) -> "Settings":
        """Load settings from a .env file (if present) and process env vars."""
        if env_path is not None:
            load_dotenv(env_path)
        else:
            default_env = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(default_env if default_env.exists() else None)

        mode_raw = (os.getenv("TRADING_MODE") or "DEMO").strip().upper()
        mode = TradingMode.LIVE if mode_raw == "LIVE" else TradingMode.DEMO

        live_confirmed = (
            os.getenv("I_UNDERSTAND_LIVE_TRADING_RISKS", "").strip()
            == _LIVE_CONFIRMATION_PHRASE
        )

        log_dir = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))

        primary = (os.getenv("SYMBOL") or "EURUSD").strip().upper()
        symbols = _parse_symbols(os.getenv("SYMBOLS"), primary)
        # Keep the primary symbol in the watchlist so the chart always has data.
        if primary not in symbols:
            symbols = [primary] + symbols

        return cls(
            mt5_login=_env_int("MT5_LOGIN"),
            mt5_password=os.getenv("MT5_PASSWORD"),
            mt5_server=os.getenv("MT5_SERVER"),
            mt5_terminal_path=os.getenv("MT5_TERMINAL_PATH"),
            mode=mode,
            live_confirmed=live_confirmed,
            kill_switch=_env_bool("KILL_SWITCH", False),
            symbol=primary,
            symbols=symbols,
            timeframe=(os.getenv("TIMEFRAME") or "M5").strip().upper(),
            scan_mode=(os.getenv("SCAN_MODE") or "best").strip().lower(),
            scan_top_n=_env_int("SCAN_TOP_N", 3) or 3,
            scan_min_score=_env_float("SCAN_MIN_SCORE", 0.0),
            htf_timeframe=(os.getenv("HTF_TIMEFRAME") or "M15").strip().upper(),
            require_htf_align=_env_bool("REQUIRE_HTF_ALIGN", True),
            ema_fast=_env_int("EMA_FAST", 9) or 9,
            ema_slow=_env_int("EMA_SLOW", 21) or 21,
            rsi_period=_env_int("RSI_PERIOD", 14) or 14,
            rsi_overbought=_env_float("RSI_OVERBOUGHT", 70.0),
            rsi_oversold=_env_float("RSI_OVERSOLD", 30.0),
            rsi_mode=(os.getenv("RSI_MODE") or "confirm").strip().lower(),
            rsi_midline=_env_float("RSI_MIDLINE", 50.0),
            atr_period=_env_int("ATR_PERIOD", 14) or 14,
            adx_period=_env_int("ADX_PERIOD", 14) or 14,
            adx_min=_env_float("ADX_MIN", 20.0),
            require_adx=_env_bool("REQUIRE_ADX", True),
            atr_multiplier=_env_float("ATR_MULTIPLIER", 1.5),
            risk_reward=_env_float("RISK_REWARD", 1.5),
            ride_trend_after_tp=_env_bool("RIDE_TREND_AFTER_TP", True),
            session_filter=_env_bool("SESSION_FILTER", True),
            # NB: don't use "or" defaults here — hour 0 (midnight UTC) is valid and
            # would be wrongly coerced to the default. _env_int returns the int default
            # when unset, so the value is always an int.
            session_start_hour=_env_int("SESSION_START_HOUR", 7),
            session_end_hour=_env_int("SESSION_END_HOUR", 16),
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_risk_per_trade=_env_float("MAX_RISK_PER_TRADE", 0.02),
            # Default is the conservative, documented 1% FIXED model. "tiered" must
            # be opted into explicitly (it scales risk up on small accounts and is
            # validated against hard caps in assert_safe_to_run).
            risk_mode=(os.getenv("RISK_MODE") or "fixed").strip().lower(),
            max_daily_loss=_env_float("MAX_DAILY_LOSS", 0.03),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 1) or 1,
            max_total_drawdown=_env_float("MAX_TOTAL_DRAWDOWN", 0.15),
            max_consecutive_losses=_env_int("MAX_CONSECUTIVE_LOSSES", 6) or 0,
            max_per_currency=_env_int("MAX_PER_CURRENCY", 1) or 1,
            max_spread_points=_env_int("MAX_SPREAD_POINTS", 0) or 0,
            trail_after_tp=_env_bool("TRAIL_AFTER_TP", True),
            trail_atr_mult=_env_float("TRAIL_ATR_MULT", 1.5),
            partial_tp_enabled=_env_bool("PARTIAL_TP_ENABLED", True),
            partial_tp_fraction=_env_float("PARTIAL_TP_FRACTION", 0.5),
            partial_tp_r=_env_float("PARTIAL_TP_R", 1.0),
            lock_profit_r=_env_float("LOCK_PROFIT_R", 0.8),
            max_bars_in_trade=_env_int("MAX_BARS_IN_TRADE", 0) or 0,
            history_bars=_env_int("HISTORY_BARS", 600) or 600,
            magic_number=_env_int("MAGIC_NUMBER", 250531) or 250531,
            deviation_points=_env_int("DEVIATION_POINTS", 20) or 20,
            reconcile_lookback_days=_env_int("RECONCILE_LOOKBACK_DAYS", 7) or 7,
            broker_utc_offset_hours=(
                None if (os.getenv("BROKER_UTC_OFFSET_HOURS") or "").strip() == ""
                else _env_float("BROKER_UTC_OFFSET_HOURS", 0.0)),
            log_dir=Path(log_dir),
        )

    def assert_safe_to_run(self) -> None:
        """Raise if the configuration is in an unsafe / contradictory state."""
        if self.live_misconfigured:
            raise RuntimeError(
                "TRADING_MODE=LIVE but I_UNDERSTAND_LIVE_TRADING_RISKS is not set to "
                f"'{_LIVE_CONFIRMATION_PHRASE}'. Refusing to start. "
                "Set TRADING_MODE=DEMO (recommended) or provide the exact confirmation phrase."
            )
        if self.ema_fast >= self.ema_slow:
            raise RuntimeError(
                f"ema_fast ({self.ema_fast}) must be < ema_slow ({self.ema_slow})."
            )
        if self.risk_mode not in {"fixed", "tiered"}:
            raise RuntimeError(
                f"risk_mode ({self.risk_mode}) must be 'fixed' or 'tiered'."
            )
        if self.risk_mode == "tiered":
            # In tiered mode the effective risk comes from the tier TABLE, not from
            # risk_per_trade / max_daily_loss. Validate every tier so an aggressive
            # (or typo'd) table can never silently run. The caps are higher than the
            # fixed-mode ceilings because tiered is an explicit, eyes-open choice —
            # but they still bound the absolute blast radius.
            from .risk import DEFAULT_RISK_TIERS  # local import avoids load-time cycle
            tier_risk_cap = 0.25   # max fraction of equity risked on a single trade
            tier_daily_cap = 0.50  # max fraction of start-of-day equity lost in a day
            for tier in DEFAULT_RISK_TIERS:
                rpt = tier.get("risk_per_trade", 0.0)
                mdl = tier.get("max_daily_loss", 0.0)
                if not (0 < rpt <= tier_risk_cap):
                    raise RuntimeError(
                        f"tiered risk_per_trade ({rpt}) for min_equity "
                        f"{tier.get('min_equity')} outside (0, {tier_risk_cap}]."
                    )
                if not (0 < mdl <= tier_daily_cap):
                    raise RuntimeError(
                        f"tiered max_daily_loss ({mdl}) for min_equity "
                        f"{tier.get('min_equity')} outside (0, {tier_daily_cap}]."
                    )
        else:  # fixed
            if not (0 < self.risk_per_trade <= 0.05):
                raise RuntimeError(
                    f"risk_per_trade ({self.risk_per_trade}) outside sane range (0, 0.05]."
                )
            if not (0 < self.max_daily_loss <= 0.20):
                raise RuntimeError(
                    f"max_daily_loss ({self.max_daily_loss}) outside sane range (0, 0.20]."
                )
        if self.scan_mode not in {"best", "top_n", "any"}:
            raise RuntimeError(
                f"scan_mode ({self.scan_mode}) must be one of: best, top_n, any."
            )
        if self.rsi_mode not in {"confirm", "filter"}:
            raise RuntimeError(
                f"rsi_mode ({self.rsi_mode}) must be 'confirm' or 'filter'."
            )
        if not (0.0 <= self.adx_min <= 100.0):
            raise RuntimeError(
                f"adx_min ({self.adx_min}) must be between 0 and 100."
            )
        if not (0.0 <= self.max_total_drawdown <= 1.0):
            raise RuntimeError(
                f"max_total_drawdown ({self.max_total_drawdown}) must be between 0 and 1."
            )
        if self.max_consecutive_losses < 0:
            raise RuntimeError("max_consecutive_losses cannot be negative.")
        if self.max_per_currency < 1:
            raise RuntimeError("max_per_currency must be at least 1.")
        if self.max_spread_points < 0:
            raise RuntimeError("max_spread_points cannot be negative.")
        if self.trail_atr_mult <= 0:
            raise RuntimeError("trail_atr_mult must be positive.")
        if not (0.0 < self.partial_tp_fraction < 1.0):
            raise RuntimeError(
                f"partial_tp_fraction ({self.partial_tp_fraction}) must be between 0 and 1 (exclusive)."
            )
        if self.partial_tp_r <= 0:
            raise RuntimeError("partial_tp_r must be positive.")
        if self.lock_profit_r < 0:
            raise RuntimeError("lock_profit_r cannot be negative.")
        if self.max_bars_in_trade < 0:
            raise RuntimeError("max_bars_in_trade cannot be negative.")
        if not (0 <= self.session_start_hour <= 23):
            raise RuntimeError(
                f"session_start_hour ({self.session_start_hour}) must be 0-23."
            )
        if not (1 <= self.session_end_hour <= 24):
            raise RuntimeError(
                f"session_end_hour ({self.session_end_hour}) must be 1-24."
            )

    def banner(self) -> str:
        """One-line human summary for the log header."""
        mode = "LIVE" if self.is_live else "DEMO"
        scope = (f"{len(self.symbols)} symbols (scan:{self.scan_mode})"
                 if len(self.symbols) > 1 else self.symbol)
        # In tiered mode the per-trade / daily-stop fractions are equity-dependent,
        # so don't print a single misleading number — say "tiered" instead.
        if self.risk_mode == "tiered":
            risk_txt = "risk TIERED (scales by equity), "
        else:
            risk_txt = (f"risk {self.risk_per_trade:.1%}/trade, "
                        f"daily-stop {self.max_daily_loss:.1%}, ")
        return (
            f"[{mode}] {scope} {self.timeframe} | "
            f"EMA {self.ema_fast}/{self.ema_slow} RSI{self.rsi_period} "
            f"ATR{self.atr_period}x{self.atr_multiplier} | "
            f"{risk_txt}"
            f"R:R 1:{self.risk_reward} | kill_switch={self.kill_switch}"
        )


if __name__ == "__main__":
    s = Settings.load()
    s.assert_safe_to_run()
    print(s.banner())
