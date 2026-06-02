"""Module 2 — Data Layer.

Wraps the official ``MetaTrader5`` library to provide a clean interface:
  * connect() / shutdown()        — manage the terminal session.
  * get_candles(symbol, tf, n)    — tidy OHLCV DataFrame, UTC index, CLOSED bars.
  * get_tick(symbol)              — latest bid/ask/spread.
  * account_info()                — equity, balance, currency.
  * symbol_spec(symbol)           — contract size / volume step / digits for sizing.

IMPORTANT: get_candles drops the still-forming (most recent) bar so the strategy
only ever sees CLOSED candles. This is the second half of the look-ahead defense
(the first being causal indicators).

MetaTrader5 only installs/runs on Windows with the MT5 terminal present. On
other platforms the import is stubbed so the rest of the package stays importable
for offline unit tests; calling any live function then raises a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:  # pragma: no cover - platform dependent
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False

from .config import Settings


# Map our string timeframes to MT5 constants lazily (mt5 may be absent offline).
def _timeframe_const(tf: str):
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 is not available on this platform.")
    table = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    key = tf.strip().upper()
    if key not in table:
        raise ValueError(f"Unsupported timeframe '{tf}'. Known: {sorted(table)}")
    return table[key]


@dataclass(frozen=True)
class Tick:
    time: datetime
    bid: float
    ask: float
    spread: float  # ask - bid, in price terms


@dataclass(frozen=True)
class Account:
    login: int
    balance: float
    equity: float
    currency: str
    server: str
    trade_mode: str  # "DEMO" / "REAL" / "CONTEST" as reported by the broker


def _require_mt5() -> None:
    if not _MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 library not available. Install on Windows with the MT5 "
            "terminal present (`uv sync` on Windows). Live data/trading is "
            "Windows-only; offline you can still run the unit tests."
        )


class DataFeed:
    """Owns the MT5 session lifecycle and exposes data accessors."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._connected = False
        self._server_offset_sec = None  # broker-server -> true-UTC seconds (detected lazily)

    # --- Session ---------------------------------------------------------
    def connect(self) -> Account:
        _require_mt5()
        kwargs = {}
        if self.s.mt5_terminal_path:
            kwargs["path"] = self.s.mt5_terminal_path
        if self.s.mt5_login:
            kwargs.update(
                login=self.s.mt5_login,
                password=self.s.mt5_password,
                server=self.s.mt5_server,
            )
        if not mt5.initialize(**kwargs):
            code, msg = mt5.last_error()
            raise ConnectionError(f"MT5 initialize() failed: ({code}) {msg}")
        self._connected = True

        # Ensure the symbol is visible in Market Watch.
        if not mt5.symbol_select(self.s.symbol, True):
            raise ConnectionError(f"Could not select symbol {self.s.symbol} in Market Watch.")

        acct = self.account_info()
        # Safety: refuse to proceed on a real account unless explicitly allowed.
        if acct.trade_mode == "REAL" and not self.s.is_live:
            self.shutdown()
            raise RuntimeError(
                "Connected terminal is a REAL/LIVE account but the bot is in DEMO "
                "mode. Refusing to continue. Use a demo account."
            )
        return acct

    def shutdown(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False

    def ensure_symbol(self, symbol: str) -> bool:
        """Make ``symbol`` visible in Market Watch so its data can be read.

        Returns False if the broker does not offer it (so the scanner can skip
        it silently). Caches successes to avoid re-selecting every cycle.
        """
        if not _MT5_AVAILABLE:
            return False
        try:
            if not hasattr(self, "_selected"):
                self._selected = set()
            if symbol in self._selected:
                return True
            if mt5.symbol_info(symbol) is None:
                return False
            if not mt5.symbol_select(symbol, True):
                return False
            self._selected.add(symbol)
            return True
        except Exception:
            return False

    # --- Broker time-zone correction -------------------------------------
    def _utc_offset_sec(self) -> int:
        """Seconds to SUBTRACT from broker timestamps to get true UTC.

        Many brokers (incl. MetaQuotes-Demo) stamp candles/ticks in their server
        timezone (often ~UTC+2/+3), not real UTC. We detect the offset once by
        comparing the broker's last tick time to the real wall-clock UTC, rounded
        to the nearest 15 minutes. A manual override (BROKER_UTC_OFFSET_HOURS) wins
        if set. Returns 0 on any failure (safe no-op)."""
        if self.s.broker_utc_offset_hours is not None:
            return int(round(self.s.broker_utc_offset_hours * 3600))
        if self._server_offset_sec is not None:
            return self._server_offset_sec
        off = 0
        try:
            t = mt5.symbol_info_tick(self.s.symbol)
            if t is not None and getattr(t, "time", 0):
                real = datetime.now(timezone.utc).timestamp()
                raw = t.time - real
                # Real broker offsets are whole/half hours, so round to 30 min.
                candidate = int(round(raw / 1800.0) * 1800)
                # Sanity gate: a fresh tick should be within the plausible server
                # offset band (~UTC-12..+14). A value outside that almost always
                # means the tick is STALE (market closed) — don't trust it; using a
                # bogus offset would mis-stamp every candle's UTC time. Fall back to
                # 0 and let BROKER_UTC_OFFSET_HOURS be set explicitly if needed.
                if -12 * 3600 <= candidate <= 14 * 3600:
                    off = candidate
                else:
                    off = 0
        except Exception:
            off = 0
        self._server_offset_sec = off
        return off

    # --- Data ------------------------------------------------------------
    def get_candles(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the last ``n`` CLOSED candles as a tidy UTC-indexed DataFrame.

        Columns: open, high, low, close, volume. The most recent (still-forming)
        bar is dropped so downstream code only sees closed candles.
        """
        _require_mt5()
        symbol = symbol or self.s.symbol
        timeframe = timeframe or self.s.timeframe
        n = n or self.s.history_bars

        tf = _timeframe_const(timeframe)
        # Fetch n+1 starting at the current forming bar (index 0), then drop it.
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n + 1)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(f"No rates for {symbol} {timeframe}: ({code}) {msg}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        off = self._utc_offset_sec()
        if off:
            df["time"] = df["time"] - pd.Timedelta(seconds=off)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        # Drop the last (forming) bar → closed candles only.
        if len(df) > 0:
            df = df.iloc[:-1]
        return df.tail(n)

    def get_tick(self, symbol: Optional[str] = None) -> Tick:
        _require_mt5()
        symbol = symbol or self.s.symbol
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"No tick for {symbol}: ({code}) {msg}")
        off = self._utc_offset_sec()
        return Tick(
            time=datetime.fromtimestamp(t.time - off, tz=timezone.utc),
            bid=t.bid,
            ask=t.ask,
            spread=round(t.ask - t.bid, 8),
        )

    def account_info(self) -> Account:
        _require_mt5()
        a = mt5.account_info()
        if a is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"account_info() failed: ({code}) {msg}")
        mode_map = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
        return Account(
            login=a.login,
            balance=a.balance,
            equity=a.equity,
            currency=a.currency,
            server=a.server,
            trade_mode=mode_map.get(a.trade_mode, str(a.trade_mode)),
        )

    def symbol_spec(self, symbol: Optional[str] = None):
        """Return a risk.SymbolSpec populated from MT5 symbol_info."""
        from .risk import SymbolSpec  # local import to avoid cycle at module load

        _require_mt5()
        symbol = symbol or self.s.symbol
        info = mt5.symbol_info(symbol)
        if info is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"symbol_info({symbol}) failed: ({code}) {msg}")
        return SymbolSpec(
            contract_size=getattr(info, "trade_contract_size", 100_000.0),
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            digits=info.digits,
            point=info.point,
        )

    def __enter__(self) -> "DataFeed":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def _check_cli() -> None:
    """`python -m bot.data --check` — Phase 1 acceptance: print tick + last candles."""
    settings = Settings.load()
    settings.assert_safe_to_run()
    print(settings.banner())
    feed = DataFeed(settings)
    acct = feed.connect()
    try:
        print(f"Account: login={acct.login} {acct.trade_mode} "
              f"equity={acct.equity} {acct.currency} @ {acct.server}")
        tick = feed.get_tick()
        print(f"Tick {tick.time.isoformat()}  bid={tick.bid} ask={tick.ask} spread={tick.spread}")
        candles = feed.get_candles(n=10)
        print(f"Last {len(candles)} CLOSED {settings.timeframe} candles for {settings.symbol}:")
        print(candles.to_string())
    finally:
        feed.shutdown()


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check_cli()
    else:
        print("Usage: python -m bot.data --check")
