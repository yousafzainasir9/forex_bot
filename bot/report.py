"""Performance reporting over the closed-trade ledger (trades.csv).

Reads the ledger written by the monitor and produces filtered performance
summaries over a chosen time window. Pure pandas — no MT5 dependency — so it
runs anywhere and is fully unit-tested.

Supported periods (filter on each trade's CLOSE time, UTC):
  today     - since 00:00 UTC today
  yesterday - the previous UTC calendar day only
  2d / 3d   - rolling last N days (N*24h back from now)
  week / 7d - rolling last 7 days
  month/30d - rolling last 30 days
  all       - no time filter
  custom    - explicit --from / --to (YYYY-MM-DD, inclusive)

CLI:
  uv run python -m bot.report --period today
  uv run python -m bot.report --period 3d
  uv run python -m bot.report --period week
  uv run python -m bot.report --period month
  uv run python -m bot.report --period custom --from 2026-05-01 --to 2026-05-15
  uv run python -m bot.report --period all --csv           # also dump filtered rows
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


PERIOD_HELP = "today | yesterday | 2d | 3d | week | month | all | custom"


def load_trades(csv_path: str | Path) -> pd.DataFrame:
    """Load trades.csv into a DataFrame with parsed UTC close/open times."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ("close_time_utc", "open_time_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("pnl", "commission", "swap", "r_multiple", "lots", "entry", "exit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _dedupe_by_position(df)


def _dedupe_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows for the same position_id, keeping the most complete
    one. The bot and the web app can both reconcile the same closed position from
    separate processes; their in-memory dedup sets can't coordinate across
    processes, so a position can occasionally land in the CSV twice (once full,
    once with blank stop/target/reason). We keep the richest row per position so
    stats and the table never double-count."""
    if df.empty or "position_id" not in df.columns:
        return df
    pid = pd.to_numeric(df["position_id"], errors="coerce")
    has_pid = pid.notna() & (pid != 0)
    if not has_pid.any():
        return df
    work = df.copy()
    work["_pid"] = pid
    # completeness score: prefer rows that carry a real R-multiple / stop-loss.
    score = pd.Series(0.0, index=work.index)
    if "r_multiple" in work.columns:
        score = score + work["r_multiple"].abs().fillna(0)
    if "stop_loss" in work.columns:
        score = score + (pd.to_numeric(work["stop_loss"], errors="coerce").fillna(0) != 0).astype(float)
    work["_score"] = score
    keep = work[has_pid].sort_values("_score", ascending=False).drop_duplicates("_pid", keep="first")
    result = pd.concat([keep, work[~has_pid]], ignore_index=True)
    return result.drop(columns=["_pid", "_score"], errors="ignore")


def filter_period(
    df: pd.DataFrame,
    period: str,
    *,
    now: Optional[datetime] = None,
    frm: Optional[str] = None,
    to: Optional[str] = None,
    time_col: str = "close_time_utc",
) -> pd.DataFrame:
    """Filter trades by close time according to ``period``.

    ``now`` defaults to the current UTC time (injectable for tests).
    """
    if df.empty or time_col not in df.columns:
        return df
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    t = df[time_col]

    p = (period or "all").lower()
    if p == "all":
        return df
    if p == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return df[t >= start]
    if p == "yesterday":
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_y = start_today - timedelta(days=1)
        return df[(t >= start_y) & (t < start_today)]
    if p in ("week", "7d"):
        return df[t >= now - timedelta(days=7)]
    if p in ("month", "30d"):
        return df[t >= now - timedelta(days=30)]
    if p.endswith("d") and p[:-1].isdigit():
        return df[t >= now - timedelta(days=int(p[:-1]))]
    if p == "custom":
        out = df
        if frm:
            start = pd.Timestamp(frm, tz="UTC")
            out = out[out[time_col] >= start]
        if to:
            end = pd.Timestamp(to, tz="UTC") + pd.Timedelta(days=1)
            out = out[out[time_col] < end]
        return out
    raise ValueError(f"Unknown period '{period}'. Use: {PERIOD_HELP}")


@dataclass
class Summary:
    period: str
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_r: float
    expectancy: float        # average net P&L per trade
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    total_commission: float
    total_swap: float

    def render(self) -> str:
        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        verdict = "PROFIT" if self.net_pnl > 0 else ("LOSS" if self.net_pnl < 0 else "FLAT")
        lines = [
            f"Performance report - {self.period}",
            "=" * 52,
            f"  Result:            {verdict}  (net P&L {self.net_pnl:+.2f})",
            f"  Trades:            {self.n_trades}  ({self.wins}W / {self.losses}L)",
            f"  Win rate:          {self.win_rate:.1%}",
            f"  Profit factor:     {pf}",
            f"  Expectancy/trade:  {self.expectancy:+.2f}",
            f"  Avg R-multiple:    {self.avg_r:+.2f}",
            f"  Gross profit/loss: {self.gross_profit:+.2f} / {-self.gross_loss:+.2f}",
            f"  Best / worst:      {self.best_trade:+.2f} / {self.worst_trade:+.2f}",
            f"  Avg win / loss:    {self.avg_win:+.2f} / {self.avg_loss:+.2f}",
            f"  Max drawdown:      {self.max_drawdown:.2f}",
            f"  Commission / swap: {self.total_commission:+.2f} / {self.total_swap:+.2f}",
        ]
        return "\n".join(lines)


def _max_drawdown(pnls: pd.Series) -> float:
    cum = pnls.cumsum()
    peak = cum.cummax()
    return float((peak - cum).max()) if len(cum) else 0.0


def summarize(df: pd.DataFrame, period_label: str = "all") -> Summary:
    """Compute performance metrics over an (already-filtered) trade frame."""
    if df is None or df.empty or "pnl" not in df.columns:
        return Summary(period_label, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Drawdown / equity curve are only meaningful in chronological order, so sort
    # by close time when available (independent of how rows sit in the CSV).
    if "close_time_utc" in df.columns:
        df = df.sort_values("close_time_utc")
    pnl = df["pnl"].fillna(0.0)
    # Only strictly-positive P&L is a win; break-even trades (pnl==0) are not counted
    # as wins so the win rate isn't inflated.
    wins_mask = pnl > 0
    wins = df[wins_mask]
    losses = df[~wins_mask]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    n = int(len(df))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_r = float(df["r_multiple"].fillna(0).mean()) if "r_multiple" in df.columns else 0.0
    comm = float(df["commission"].fillna(0).sum()) if "commission" in df.columns else 0.0
    swap = float(df["swap"].fillna(0).sum()) if "swap" in df.columns else 0.0

    return Summary(
        period=period_label,
        n_trades=n,
        wins=int(len(wins)),
        losses=int(len(losses)),
        win_rate=(len(wins) / n) if n else 0.0,
        net_pnl=round(float(pnl.sum()), 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        profit_factor=pf,
        avg_r=round(avg_r, 3),
        expectancy=round(float(pnl.mean()), 2) if n else 0.0,
        best_trade=round(float(pnl.max()), 2) if n else 0.0,
        worst_trade=round(float(pnl.min()), 2) if n else 0.0,
        avg_win=round(float(wins["pnl"].mean()), 2) if not wins.empty else 0.0,
        avg_loss=round(float(losses["pnl"].mean()), 2) if not losses.empty else 0.0,
        max_drawdown=round(_max_drawdown(pnl), 2),
        total_commission=round(comm, 2),
        total_swap=round(swap, 2),
    )


def build_report(
    csv_path: str | Path,
    period: str = "all",
    *,
    frm: Optional[str] = None,
    to: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[Summary, pd.DataFrame]:
    df = load_trades(csv_path)
    filtered = filter_period(df, period, now=now, frm=frm, to=to)
    label = period if period != "custom" else f"custom {frm or '...'} -> {to or '...'}"
    return summarize(filtered, label), filtered


def _default_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "logs" / "trades.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forex bot performance report.")
    parser.add_argument("--period", default="all", help=PERIOD_HELP)
    parser.add_argument("--from", dest="frm", default=None, help="custom start YYYY-MM-DD")
    parser.add_argument("--to", dest="to", default=None, help="custom end YYYY-MM-DD (inclusive)")
    parser.add_argument("--file", default=str(_default_csv()), help="path to trades.csv")
    parser.add_argument("--csv", action="store_true", help="also print the filtered trade rows")
    args = parser.parse_args(argv)

    summary, rows = build_report(args.file, args.period, frm=args.frm, to=args.to)
    print(summary.render())
    if args.csv and not rows.empty:
        cols = [c for c in ["close_time_utc", "symbol", "side", "lots", "entry",
                            "exit", "pnl", "r_multiple"] if c in rows.columns]
        print("\nTrades:")
        print(rows[cols].to_string(index=False))
    elif args.csv:
        print("\n(no trades in this period)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
