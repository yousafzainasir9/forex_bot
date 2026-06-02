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

    # --- Strategy params ---
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    atr_period: int
    atr_multiplier: float
    risk_reward: float
    ride_trend_after_tp: bool  # once target hit, hold & trail on closes instead of closing at TP

    # --- Risk params ---
    risk_per_trade: float       # fraction of equity, e.g. 0.01 = 1%
    max_risk_per_trade: float   # hard ceiling: veto a trade whose min-lot risk exceeds this
    risk_mode: str              # 'tiered' (scale risk by equity) or 'fixed'
    max_daily_loss: float       # fraction of start-of-day equity, e.g. 0.03 = 3%
    max_open_positions: int

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
            ema_fast=_env_int("EMA_FAST", 9) or 9,
            ema_slow=_env_int("EMA_SLOW", 21) or 21,
            rsi_period=_env_int("RSI_PERIOD", 14) or 14,
            rsi_overbought=_env_float("RSI_OVERBOUGHT", 70.0),
            rsi_oversold=_env_float("RSI_OVERSOLD", 30.0),
            atr_period=_env_int("ATR_PERIOD", 14) or 14,
            atr_multiplier=_env_float("ATR_MULTIPLIER", 1.5),
            risk_reward=_env_float("RISK_REWARD", 1.5),
            ride_trend_after_tp=_env_bool("RIDE_TREND_AFTER_TP", True),
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_risk_per_trade=_env_float("MAX_RISK_PER_TRADE", 0.02),
            risk_mode=(os.getenv("RISK_MODE") or "tiered").strip().lower(),
            max_daily_loss=_env_float("MAX_DAILY_LOSS", 0.03),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 1) or 1,
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

    def banner(self) -> str:
        """One-line human summary for the log header."""
        mode = "LIVE" if self.is_live else "DEMO"
        scope = (f"{len(self.symbols)} symbols (scan:{self.scan_mode})"
                 if len(self.symbols) > 1 else self.symbol)
        return (
            f"[{mode}] {scope} {self.timeframe} | "
            f"EMA {self.ema_fast}/{self.ema_slow} RSI{self.rsi_period} "
            f"ATR{self.atr_period}x{self.atr_multiplier} | "
            f"risk {self.risk_per_trade:.1%}/trade, daily-stop {self.max_daily_loss:.1%}, "
            f"R:R 1:{self.risk_reward} | kill_switch={self.kill_switch}"
        )


if __name__ == "__main__":
    s = Settings.load()
    s.assert_safe_to_run()
    print(s.banner())
