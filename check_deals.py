"""Diagnostic — show recent CLOSED deals straight from MT5.

Run (from the forex_bot folder, with MT5 open):
    uv run python check_deals.py
    uv run python check_deals.py --days 30        # widen the lookback

It prints every closed position MT5 reports in the window, with its symbol,
side, net P&L, and — crucially — the magic number. Compare that magic to the
bot's configured MAGIC_NUMBER (printed at the top): trades whose magic differs
are NOT recorded into trades.csv (the performance ledger), which is the usual
reason a completed trade is missing from the dashboard history.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from bot.config import Settings


def main() -> int:
    ap = argparse.ArgumentParser(description="Show recent closed MT5 deals.")
    ap.add_argument("--days", type=int, default=14, help="lookback window in days")
    args = ap.parse_args()

    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        print("MetaTrader5 library not available (Windows + terminal required).")
        return 1

    s = Settings.load()
    print(f"Bot magic number (only this is recorded to history): {s.magic_number}")
    print(f"Account in .env: login={s.mt5_login} server={s.mt5_server}")

    kwargs = {}
    if s.mt5_terminal_path:
        kwargs["path"] = s.mt5_terminal_path
    if s.mt5_login:
        kwargs.update(login=s.mt5_login, password=s.mt5_password, server=s.mt5_server)
    if not mt5.initialize(**kwargs):
        print("MT5 initialize() failed:", mt5.last_error())
        return 1

    try:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
        if not deals:
            print(f"\nNo deals at all in the last {args.days} days.")
            print("Either nothing has closed, or the window is too short "
                  "(try --days 60).")
            return 0

        groups: dict = defaultdict(list)
        for d in deals:
            pid = getattr(d, "position_id", 0) or 0
            groups[pid].append(d)

        # entry==1 is an OUT (closing) deal in MT5's enum.
        DEAL_ENTRY_OUT = 1
        closed = []
        for pid, ds in groups.items():
            if not pid:
                continue
            if not any(getattr(d, "entry", None) == DEAL_ENTRY_OUT for d in ds):
                continue  # still open
            pnl = sum(float(getattr(d, "profit", 0) or 0)
                      + float(getattr(d, "commission", 0) or 0)
                      + float(getattr(d, "swap", 0) or 0) for d in ds)
            magic = next((getattr(d, "magic", 0) for d in ds), 0)
            sym = next((getattr(d, "symbol", "") for d in ds), "")
            close_t = max(getattr(d, "time", 0) for d in ds)
            closed.append((close_t, pid, sym, magic, round(pnl, 2)))

        if not closed:
            print(f"\nFound {len(deals)} deals but NO fully-closed positions in "
                  f"the last {args.days} days — your trades are likely still open.")
            return 0

        closed.sort()
        bot_n = sum(1 for c in closed if c[3] == s.magic_number)
        print(f"\nClosed positions in the last {args.days} days: {len(closed)} "
              f"({bot_n} match the bot's magic {s.magic_number}, "
              f"{len(closed) - bot_n} do NOT and are skipped by history)\n")
        print(f"{'close time (UTC)':<20} {'position_id':<14} {'symbol':<10} "
              f"{'magic':<10} {'net P&L':>10}  recorded?")
        print("-" * 80)
        for close_t, pid, sym, magic, pnl in closed:
            ts = datetime.fromtimestamp(close_t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            rec = "YES" if magic == s.magic_number else "no (magic mismatch)"
            print(f"{ts:<20} {pid:<14} {sym:<10} {magic:<10} {pnl:>10.2f}  {rec}")
    finally:
        mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
