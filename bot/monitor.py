"""Module 7 — Logger & Monitor.

Records every signal, risk decision, veto reason, and order result, computes
running performance statistics, and writes a daily summary.

Outputs (under ``settings.log_dir``):
  * bot.log            — human-readable structured log (also echoed to console).
  * trades.csv         — one row per CLOSED trade (append-only, deduped by id).
  * open_trades.json   — bookkeeping for currently-open positions (for R-multiple).
  * daily_YYYYMMDD.txt — per-day summary written at rollover / shutdown.

Running stats (win rate, profit factor, R-multiples, max drawdown) are computed
from the closed-trade ledger so they survive restarts. Realized P&L is the broker
truth (deal profit + commission + swap), so SL/TP exits the broker fills on its
own are captured just like bot-initiated closes.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


TRADE_CSV_HEADER = [
    "close_time_utc", "open_time_utc", "symbol", "side", "lots",
    "entry", "exit", "stop_loss", "take_profit",
    "pnl", "commission", "swap", "r_multiple", "position_id",
    "reason_open", "reason_close",
]


@dataclass
class ClosedTrade:
    open_time_utc: str
    close_time_utc: str
    symbol: str
    side: str
    lots: float
    entry: float
    exit: float
    stop_loss: float
    take_profit: float
    pnl: float                 # NET realized P&L = profit + commission + swap
    commission: float = 0.0
    swap: float = 0.0
    r_multiple: float = 0.0    # pnl / planned-risk (0 if planned risk unknown)
    position_id: int = 0
    reason_open: str = ""
    reason_close: str = ""


@dataclass
class Stats:
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0      # stored as a positive magnitude
    sum_r: float = 0.0
    consecutive_losses: int = 0  # trailing run of losing trades (reset by any win)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def net_pnl(self) -> float:
        return self.gross_profit - self.gross_loss

    @property
    def avg_r(self) -> float:
        return self.sum_r / self.n_trades if self.n_trades else 0.0

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drop of the cumulative-PnL curve (money terms)."""
        peak = float("-inf")
        max_dd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
        return max_dd


class Monitor:
    def __init__(self, log_dir: Path, *, console: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trades_csv = self.log_dir / "trades.csv"
        self.open_trades_file = self.log_dir / "open_trades.json"
        self.stats = Stats()
        self._current_day: Optional[date] = None
        self.recorded_position_ids: set[int] = set()
        self._lock_path = self.log_dir / "trades.csv.lock"

        # --- Configure a dedicated logger (no duplicate handlers on re-init). ---
        self.log = logging.getLogger("forex_bot")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.log_dir / "bot.log", encoding="utf-8")
        fh.setFormatter(fmt)
        self.log.addHandler(fh)
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self.log.addHandler(ch)

        self._ensure_csv_header()
        self._load_existing_trades()

    # --- Setup helpers ---------------------------------------------------
    def _ensure_csv_header(self) -> None:
        if not self.trades_csv.exists():
            with self.trades_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(TRADE_CSV_HEADER)

    def _load_existing_trades(self) -> None:
        """Rebuild running stats + dedupe set from any existing trades.csv."""
        if not self.trades_csv.exists():
            return
        with self.trades_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self._fold_trade(float(row["pnl"]), float(row.get("r_multiple", 0) or 0))
                    pid = int(float(row.get("position_id", 0) or 0))
                    if pid:
                        self.recorded_position_ids.add(pid)
                except (KeyError, ValueError):
                    continue

    def _fold_trade(self, pnl: float, r_multiple: float) -> None:
        self.stats.n_trades += 1
        self.stats.sum_r += r_multiple
        # Only strictly-positive P&L counts as a win; a break-even trade (pnl==0)
        # adds nothing to gross profit/loss and is recorded on the loss side of the
        # count so the win rate isn't inflated.
        if pnl > 0:
            self.stats.wins += 1
            self.stats.gross_profit += pnl
            self.stats.consecutive_losses = 0
        else:
            self.stats.losses += 1
            self.stats.gross_loss += -pnl
            self.stats.consecutive_losses += 1
        prev = self.stats.equity_curve[-1] if self.stats.equity_curve else 0.0
        self.stats.equity_curve.append(prev + pnl)

    # --- Open-trade bookkeeping (for R-multiple on close) ----------------
    def _read_open_map(self) -> Dict[str, dict]:
        if not self.open_trades_file.exists():
            return {}
        try:
            return json.loads(self.open_trades_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_open_map(self, data: Dict[str, dict]) -> None:
        self.open_trades_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def note_open(self, position_id: int, info: dict) -> None:
        """Remember a freshly-opened position so we can compute R when it closes."""
        data = self._read_open_map()
        data[str(position_id)] = info
        self._write_open_map(data)

    def pop_open(self, position_id: int) -> Optional[dict]:
        """Fetch + remove the stored open info for a position (if any)."""
        data = self._read_open_map()
        info = data.pop(str(position_id), None)
        if info is not None:
            self._write_open_map(data)
        return info

    # --- Event logging ---------------------------------------------------
    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)

    def error(self, msg: str) -> None:
        self.log.error(msg)

    def log_signal(self, signal) -> None:
        self.log.info(f"SIGNAL {signal}")

    def log_risk(self, decision) -> None:
        tag = "APPROVE" if decision.approved else "VETO"
        self.log.info(f"RISK {tag}: {decision.reason}")

    def log_order(self, ok: bool, detail: str) -> None:
        (self.log.info if ok else self.log.error)(
            f"ORDER {'OK' if ok else 'FAIL'}: {detail}"
        )

    def already_recorded(self, position_id: int) -> bool:
        return position_id in self.recorded_position_ids

    def record_trade(self, trade: ClosedTrade) -> bool:
        """Append a closed trade to the ledger and update running stats.

        Idempotent against the CSV file *on disk* (re-read here), not just this
        process's in-memory set, so clearing history makes positions writable again
        and the bot + web app never double-record or clobber each other's rows.
        The whole read-decide-append runs under a cross-process file lock.
        """
        with self._csv_lock():
            self._resync_from_disk()
            if trade.position_id and trade.position_id in self.recorded_position_ids:
                return False
            self._ensure_csv_header()
            with self.trades_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    trade.close_time_utc, trade.open_time_utc, trade.symbol, trade.side,
                    trade.lots, trade.entry, trade.exit, trade.stop_loss, trade.take_profit,
                    trade.pnl, trade.commission, trade.swap, trade.r_multiple,
                    trade.position_id, trade.reason_open, trade.reason_close,
                ])
            self._fold_trade(trade.pnl, trade.r_multiple)
            if trade.position_id:
                self.recorded_position_ids.add(trade.position_id)
        self.log.info(
            f"TRADE CLOSED {trade.side} {trade.symbol} pnl={trade.pnl:+.2f} "
            f"R={trade.r_multiple:+.2f} | {trade.reason_close}"
        )
        return True

    def _resync_from_disk(self) -> None:
        """Rebuild stats + dedup set from the current trades.csv so they reflect
        external edits (history cleared, or rows added by another process)."""
        self.stats = Stats()
        self.recorded_position_ids = set()
        self._load_existing_trades()

    @contextlib.contextmanager
    def _csv_lock(self, timeout: float = 10.0, poll: float = 0.05):
        """Best-effort cross-process exclusive lock around trades.csv via an atomic
        O_CREAT|O_EXCL lock file. Proceeds after ``timeout`` rather than
        deadlocking, and reclaims a stale lock left by a crashed process."""
        acquired = False
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd); acquired = True; break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self._lock_path) > timeout:
                        os.unlink(self._lock_path); continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(poll)
            except OSError:
                break
        try:
            yield
        finally:
            if acquired:
                try: os.unlink(self._lock_path)
                except OSError: pass

    # --- Summaries -------------------------------------------------------
    def stats_line(self) -> str:
        s = self.stats
        pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
        return (
            f"trades={s.n_trades} win%={s.win_rate:.1%} PF={pf} "
            f"netPnL={s.net_pnl:+.2f} avgR={s.avg_r:+.2f} maxDD={s.max_drawdown:.2f}"
        )

    def daily_summary(self, equity: Optional[float] = None,
                      day: Optional[date] = None) -> Path:
        """Write a per-day summary file and return its path."""
        day = day or datetime.now(timezone.utc).date()
        path = self.log_dir / f"daily_{day:%Y%m%d}.txt"
        lines = [
            f"Daily summary - {day.isoformat()} (UTC)",
            "=" * 48,
            self.stats_line(),
        ]
        if equity is not None:
            lines.append(f"equity={equity:.2f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log.info(f"Wrote daily summary -> {path.name} | {self.stats_line()}")
        return path

    def maybe_rollover(self, equity: Optional[float] = None) -> bool:
        """Detect a UTC day change; write yesterday's summary if so."""
        today = datetime.now(timezone.utc).date()
        if self._current_day is None:
            self._current_day = today
            return False
        if today != self._current_day:
            self.daily_summary(equity=equity, day=self._current_day)
            self._current_day = today
            return True
        return False
