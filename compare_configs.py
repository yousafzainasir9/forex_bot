"""Compare the pre-review baseline against the new recommended defaults.

Runs the SAME backtest pipeline under several configurations and prints a
side-by-side table so you can see what the whipsaw filters actually do to trade
count, win rate, profit factor, expectancy and drawdown.

Why a separate script: it just wires presets into bot.backtest — no new strategy
logic — so the comparison uses the exact code the live bot runs.

Usage
-----
  # Synthetic data (quick smoke test, NOT a real edge signal):
  uv run python compare_configs.py --demo

  # Your own M5 CSV (time,open,high,low,close[,volume]):
  uv run python compare_configs.py --csv data/EURUSD_M5.csv

  # Pull history straight from MT5 (Windows + terminal running):
  uv run python compare_configs.py --symbol EURUSD --bars 8000

  # Honest out-of-sample estimate (walk-forward each preset):
  uv run python compare_configs.py --csv data/EURUSD_M5.csv --walk-forward

Costs default to a realistic ~1.2-pip spread; override with --spread/--commission.
A backtest can rule out disasters; it cannot prove a live edge. Read the numbers
as a sanity check, never a promise.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from bot.backtest import (
    CostModel,
    StrategyParams,
    fetch_mt5_history,
    load_csv,
    make_synthetic,
    run_backtest,
    walk_forward,
)


# Each preset = (label, StrategyParams overrides, run_backtest kwargs).
# Held constant across presets: EMA 9/21, ATR 14x1.5, R:R 1.5, fixed 1% risk — so
# the ONLY thing changing is the set of entry filters we added in the review.
def _presets() -> List[Tuple[str, dict, dict]]:
    return [
        (
            "Baseline (pre-review)",
            {"rsi_mode": "filter"},
            {"require_htf_align": False, "session_filter": False,
             "ride_trend_after_tp": True, "risk_mode": "fixed"},
        ),
        (
            "RSI confirm only",
            {"rsi_mode": "confirm", "rsi_midline": 50.0},
            {"require_htf_align": False, "session_filter": False,
             "ride_trend_after_tp": True, "risk_mode": "fixed"},
        ),
        (
            "+ HTF gate",
            {"rsi_mode": "confirm", "rsi_midline": 50.0},
            {"require_htf_align": True, "session_filter": False,
             "ride_trend_after_tp": True, "risk_mode": "fixed"},
        ),
        (
            "+ HTF + session",
            {"rsi_mode": "confirm", "rsi_midline": 50.0},
            {"require_htf_align": True, "session_filter": True,
             "session_start_hour": 7, "session_end_hour": 16,
             "ride_trend_after_tp": True, "risk_mode": "fixed"},
        ),
        (
            "New defaults (+ADX)",
            {"rsi_mode": "confirm", "rsi_midline": 50.0, "adx_min": 20.0},
            {"require_htf_align": True, "session_filter": True,
             "session_start_hour": 7, "session_end_hour": 16,
             "ride_trend_after_tp": True, "risk_mode": "fixed",
             "trail_after_tp": True, "trail_atr_mult": 1.5,
             "partial_tp_enabled": True, "partial_tp_fraction": 0.5,
             "partial_tp_r": 1.0, "lock_profit_r": 0.8},
        ),
    ]


def _pf(v: float) -> str:
    return "inf" if v == float("inf") else f"{v:.2f}"


def _row(label: str, s: Dict) -> str:
    return (
        f"{label:<28} {s['n_trades']:>6} {s['win_rate']*100:>6.1f}% "
        f"{_pf(s['profit_factor']):>7} {s['expectancy']:>+9.2f} "
        f"{s['avg_r']:>+7.2f} {s['max_drawdown']:>9.2f} "
        f"{s['net_pnl']:>+11.2f} {s['return_pct']:>+8.2f}%"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare baseline vs new defaults.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="OHLC CSV (time,open,high,low,close).")
    src.add_argument("--demo", action="store_true", help="use built-in synthetic data.")
    src.add_argument("--symbol", help="pull history from MT5 for this symbol (Windows).")
    p.add_argument("--timeframe", default="M5")
    p.add_argument("--bars", type=int, default=8000, help="bars to pull from MT5.")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--spread", type=float, default=0.00012, help="spread in price (0.00012 = 1.2 pips).")
    p.add_argument("--commission", type=float, default=0.0, help="commission per lot (round-turn).")
    p.add_argument("--walk-forward", action="store_true",
                   help="score each preset out-of-sample (the honest estimate).")
    p.add_argument("--is-bars", type=int, default=2000)
    p.add_argument("--oos-bars", type=int, default=500)
    args = p.parse_args(argv)

    if args.csv:
        df = load_csv(args.csv)
    elif args.symbol:
        df = fetch_mt5_history(args.symbol, args.timeframe, args.bars)
    else:
        df = make_synthetic()
        if not args.demo:
            print("(no --csv/--symbol; using synthetic --demo data — illustrative only)")

    costs = CostModel(spread_price=args.spread, commission_per_lot=args.commission)
    common = dict(starting_equity=args.equity, risk_per_trade=args.risk, costs=costs)

    mode = "walk-forward OOS" if args.walk_forward else "full-period"
    print(f"\nConfig comparison ({mode}) — {len(df)} bars, "
          f"spread {args.spread}, commission {args.commission}/lot\n")
    header = (f"{'Preset':<28} {'Trades':>6} {'Win%':>7} {'PF':>7} "
              f"{'Expect':>9} {'AvgR':>7} {'MaxDD':>9} {'NetPnL':>11} {'Return':>9}")
    print(header)
    print("-" * len(header))

    for label, sp_over, bt_over in _presets():
        base = StrategyParams(**sp_over)
        kwargs = {**common, **bt_over}
        if args.walk_forward:
            wf = walk_forward(df, is_bars=args.is_bars, oos_bars=args.oos_bars,
                              base=base, **kwargs)
            summ = wf.oos_summary
        else:
            summ = run_backtest(df, base, **kwargs).summary
        print(_row(label, summ))

    print("\nRead the LAST row (new defaults) against the FIRST (baseline). Fewer "
          "trades with higher win%/PF/expectancy = the filters are earning their keep. "
          "If a filter cuts trades but also cuts expectancy, drop it.\n"
          "Reminder: backtests flatter themselves — trust walk-forward OOS over full-period, "
          "and demo-forward over both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
