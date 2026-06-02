"""Robustness & risk-of-ruin check — the evidence gate before risking real money.

A single backtest number is nearly meaningless: it's one path out of many that the
same edge could have produced. This script:

  1. Runs WALK-FORWARD on your data using your live .env config (the honest,
     out-of-sample estimate — not an in-sample curve fitted to history).
  2. MONTE-CARLO bootstraps the out-of-sample trade sequence thousands of times to
     estimate the *distribution* of outcomes: median return, a bad-case (5th pct)
     return, the chance the system is net-negative, the drawdown you should expect,
     and an approximate risk of ruin.

What "good" looks like: a positive median AND a 5th-percentile that you could
stomach, a low probability of net loss, and a near-zero risk of ruin at your chosen
threshold. If the bad cases would wipe you out, the edge (if any) is too thin or too
volatile to fund — regardless of how nice the average looks.

Usage
-----
  uv run python validate_robustness.py --symbol EURUSD --bars 12000
  uv run python validate_robustness.py --csv data/EURUSD_M5.csv
  uv run python validate_robustness.py --demo                 # synthetic, illustrative only

Assumes trades are roughly independent (bootstrap). That's an approximation — real
trades cluster — so treat the ruin estimate as optimistic, not gospel. A backtest
can rule out disasters; it cannot prove a live edge.
"""

from __future__ import annotations

import argparse

import numpy as np

from bot.backtest import (
    CostModel, _params_from_settings, fetch_mt5_history, load_csv,
    make_synthetic, walk_forward,
)


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else 0.0


def _bt_kwargs_from_settings(equity: float, risk: float, costs: CostModel) -> dict:
    """Mirror the live .env config so the validation reflects how the bot trades."""
    kw = dict(starting_equity=equity, risk_per_trade=risk, costs=costs)
    try:
        from bot.config import Settings
        s = Settings.load()
        kw.update(
            ride_trend_after_tp=s.ride_trend_after_tp, risk_mode=s.risk_mode,
            require_htf_align=s.require_htf_align, htf_timeframe=s.htf_timeframe,
            session_filter=s.session_filter, session_start_hour=s.session_start_hour,
            session_end_hour=s.session_end_hour,
            trail_after_tp=s.trail_after_tp, trail_atr_mult=s.trail_atr_mult,
            partial_tp_enabled=s.partial_tp_enabled,
            partial_tp_fraction=s.partial_tp_fraction, partial_tp_r=s.partial_tp_r,
            lock_profit_r=s.lock_profit_r, max_bars_in_trade=s.max_bars_in_trade,
        )
    except Exception as e:  # offline / no .env -> sensible defaults
        print(f"(could not load .env, using defaults: {e})")
    return kw


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward + Monte-Carlo robustness check.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="OHLC CSV (time,open,high,low,close).")
    src.add_argument("--demo", action="store_true", help="use built-in synthetic data.")
    src.add_argument("--symbol", help="pull history from MT5 for this symbol (Windows).")
    p.add_argument("--timeframe", default="M5")
    p.add_argument("--bars", type=int, default=12000)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--spread", type=float, default=0.00012)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--is-bars", type=int, default=2000)
    p.add_argument("--oos-bars", type=int, default=500)
    p.add_argument("--sims", type=int, default=5000, help="Monte-Carlo iterations.")
    p.add_argument("--ruin", type=float, default=0.50,
                   help="ruin = losing this fraction of starting equity (default 50%).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.csv:
        df = load_csv(args.csv)
    elif args.symbol:
        df = fetch_mt5_history(args.symbol, args.timeframe, args.bars)
    else:
        df = make_synthetic(n=6000)
        if not args.demo:
            print("(no --csv/--symbol; using synthetic data — illustrative only)")

    costs = CostModel(spread_price=args.spread, commission_per_lot=args.commission)
    kw = _bt_kwargs_from_settings(args.equity, args.risk, costs)

    print(f"\nWalk-forward on {len(df)} bars "
          f"(IS={args.is_bars} / OOS={args.oos_bars} bars per fold)...")
    wf = walk_forward(df, is_bars=args.is_bars, oos_bars=args.oos_bars,
                      base=_params_from_settings(), **kw)
    s = wf.oos_summary
    print(wf.render())

    trades = wf.oos_trades
    if trades.empty or "pnl" not in trades.columns or len(trades) < 20:
        n = 0 if trades.empty else len(trades)
        print(f"\nOnly {n} out-of-sample trades — too few for a meaningful Monte-Carlo. "
              "Gather more history (more bars / more instruments) before trusting any "
              "robustness estimate.")
        return 0

    pnls = trades["pnl"].to_numpy(dtype=float)
    n = len(pnls)
    rng = np.random.default_rng(args.seed)
    ruin_level = args.ruin * args.equity

    finals = np.empty(args.sims)
    maxdds = np.empty(args.sims)
    ruined = 0
    for k in range(args.sims):
        sample = rng.choice(pnls, size=n, replace=True)  # bootstrap the trade sequence
        cum = np.cumsum(sample)
        finals[k] = cum[-1]
        peak = np.maximum.accumulate(cum)
        maxdds[k] = float(np.max(peak - cum))
        if cum.min() <= -ruin_level:
            ruined += 1

    ret = 100.0 * finals / args.equity
    print("\n" + "=" * 64)
    print(f"Monte-Carlo robustness ({args.sims} bootstraps of {n} OOS trades)")
    print("=" * 64)
    print(f"  Median return:           {np.median(ret):+.2f}%")
    print(f"  Bad case (5th pct):      {_pct(ret, 5):+.2f}%")
    print(f"  Good case (95th pct):    {_pct(ret, 95):+.2f}%")
    print(f"  P(net loss):             {100.0 * np.mean(finals < 0):.1f}%")
    print(f"  Median max drawdown:     {np.median(maxdds):.2f} "
          f"({100.0 * np.median(maxdds) / args.equity:.1f}% of equity)")
    print(f"  Worst-case DD (95th pct):{_pct(maxdds, 95):.2f} "
          f"({100.0 * _pct(maxdds, 95) / args.equity:.1f}% of equity)")
    print(f"  Risk of ruin (-{args.ruin:.0%}):     {100.0 * ruined / args.sims:.2f}%")
    print("\nFund it only if: median return positive, the 5th-percentile is survivable,"
          "\nP(net loss) is low, and risk of ruin is ~0. Then STILL forward-test on demo"
          "\nfor weeks and reconcile against the broker statement before committing money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
