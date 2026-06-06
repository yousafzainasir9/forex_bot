"""CTA-style diversified BASKET walk-forward.

Real systematic trend-followers (CTAs) don't rely on a strong edge in any single
market — they apply a SMALL edge across MANY uncorrelated instruments, so the
per-market noise diversifies away and a smooth, positive portfolio curve emerges
even when each market alone is only marginal.

This script tests exactly that idea with the SAME honest walk-forward we've used
all along: for each instrument it fetches history from MT5, runs the walk-forward
(in-sample param selection -> out-of-sample scoring), then AGGREGATES every
instrument's out-of-sample trades into one portfolio.

Aggregation is done in R-MULTIPLES (P&L / money-at-risk), not dollars, because the
backtester uses generic FX contract specs offline and dollar P&L would be wrong
for metals/indices. R normalises every trade to "multiples of the risk taken", so
instruments with very different pip values combine correctly.

Usage (Windows, MT5 running & logged into the demo):
    uv run python basket_backtest.py
    uv run python basket_backtest.py --timeframe H1 --htf H4 --bars 20000 \
        --is-bars 3000 --oos-bars 750 --spread 0.00012
    uv run python basket_backtest.py --symbols EURUSD,GBPUSD,XAUUSD,US500 --spread 0.0002

This is research, not a promise. A positive aggregate that SURVIVES a realistic
spread and shows a smoother curve than any single market is the signal worth
chasing; anything else means the basket has no edge either.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from bot.backtest import (walk_forward, fetch_mt5_history, _params_from_settings,
                          CostModel)

# A deliberately DIVERSIFIED default basket: FX majors that don't all move
# together, plus a metal and (if your broker offers them) indices. The more
# uncorrelated the markets, the stronger the diversification effect.
DEFAULT_BASKET = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
                  "USDCAD", "NZDUSD", "XAUUSD"]


def _pf(r: pd.Series) -> float:
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return (gp / gl) if gl > 0 else float("inf")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CTA-style diversified basket walk-forward.")
    p.add_argument("--symbols", default=",".join(DEFAULT_BASKET),
                   help="comma/space separated instruments (default: a diversified basket).")
    p.add_argument("--timeframe", default="H1", help="trading timeframe (default H1).")
    p.add_argument("--htf", default="H4",
                   help="higher timeframe for the trend gate (must be SLOWER than --timeframe; default H4).")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--is-bars", type=int, default=3000)
    p.add_argument("--oos-bars", type=int, default=750)
    p.add_argument("--spread", type=float, default=0.00012, help="spread in price (0.00012 = 1.2 pips).")
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--risk", type=float, default=0.01, help="risk fraction per trade for the portfolio curve.")
    p.add_argument("--no-htf-gate", action="store_true", help="disable the higher-timeframe gate.")
    args = p.parse_args(argv)

    # Reuse the live exit/risk configuration from .env so the basket trades the
    # same way the bot actually would; only the instruments and timeframe change.
    try:
        from bot.config import Settings
        s = Settings.load()
    except Exception as e:
        print(f"Could not load settings: {e}")
        return 1

    costs = CostModel(spread_price=args.spread, commission_per_lot=args.commission)
    bt = dict(
        starting_equity=10_000.0, risk_per_trade=args.risk, costs=costs,
        ride_trend_after_tp=s.ride_trend_after_tp, risk_mode="fixed",
        require_htf_align=(not args.no_htf_gate and s.require_htf_align),
        htf_timeframe=args.htf,
        session_filter=s.session_filter,
        session_start_hour=s.session_start_hour, session_end_hour=s.session_end_hour,
        trail_after_tp=s.trail_after_tp, trail_atr_mult=s.trail_atr_mult,
        partial_tp_enabled=s.partial_tp_enabled, partial_tp_fraction=s.partial_tp_fraction,
        partial_tp_r=s.partial_tp_r, lock_profit_r=s.lock_profit_r,
        ride_stall_atr_frac=s.ride_stall_atr_frac, max_bars_in_trade=s.max_bars_in_trade,
    )
    base = _params_from_settings()
    symbols = [x.strip().upper()
               for chunk in args.symbols.split(",")
               for x in chunk.replace(";", " ").split() if x.strip()]

    gate = "off" if (args.no_htf_gate or not s.require_htf_align) else args.htf
    print(f"CTA BASKET walk-forward | {args.timeframe} | HTF gate: {gate} | "
          f"spread {args.spread} ({args.spread*10000:.1f} pip) | {len(symbols)} symbols")
    print("=" * 78)
    print(f"  {'symbol':8} {'trades':>7} {'win%':>6} {'PF':>6} {'sumR':>8}   overfit$ (hindsight)")
    print("-" * 78)

    all_oos = []
    for sym in symbols:
        try:
            df = fetch_mt5_history(sym, args.timeframe, args.bars)
        except Exception as e:
            print(f"  {sym:8}  SKIP — no data ({e})")
            continue
        try:
            wf = walk_forward(df, is_bars=args.is_bars, oos_bars=args.oos_bars, base=base, **bt)
        except Exception as e:
            print(f"  {sym:8}  ERROR — {e}")
            continue
        oos = wf.oos_trades
        n = len(oos)
        if n == 0:
            print(f"  {sym:8} {0:>7} {'-':>6} {'-':>6} {0.0:>8.1f}   {wf.full_in_sample.get('net_pnl', 0.0):+8.0f}")
            continue
        win = 100.0 * (oos["pnl"] > 0).mean()
        sumr = oos["r_multiple"].sum()
        pf = _pf(oos["r_multiple"])
        overfit = wf.full_in_sample.get("net_pnl", 0.0)
        print(f"  {sym:8} {n:>7} {win:>6.1f} {pf:>6.2f} {sumr:>+8.1f}   {overfit:+8.0f}")
        o = oos.copy()
        o["symbol"] = sym
        all_oos.append(o)

    if not all_oos:
        print("\nNo out-of-sample trades produced. Check the timeframe/HTF and broker symbols.")
        return 0

    allt = pd.concat(all_oos, ignore_index=True)
    allt["ct"] = pd.to_datetime(allt["close_time_utc"], errors="coerce", utc=True)
    allt = allt.sort_values("ct").reset_index(drop=True)

    n = len(allt)
    win = 100.0 * (allt["pnl"] > 0).mean()
    pf = _pf(allt["r_multiple"])
    exp = allt["r_multiple"].mean()
    total_r = allt["r_multiple"].sum()

    # Portfolio equity curve in R: each closed trade risks `--risk` of equity and
    # returns r * risk_frac. Sequenced by close time across ALL instruments. This
    # is a conservative view (it ignores the extra smoothing from positions held
    # concurrently in uncorrelated markets), but it answers the key question:
    # is the AGGREGATE edge positive, and is the drawdown tamer than any single market?
    eq = 10_000.0
    peak = eq
    mdd = 0.0
    for r in allt["r_multiple"]:
        eq *= (1.0 + args.risk * float(r))
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)

    print("=" * 78)
    print(f"AGGREGATE OOS (R-normalised across basket): trades={n}  win%={win:.1f}  "
          f"PF={pf:.2f}  expectancy={exp:+.3f} R/trade")
    print(f"  total R={total_r:+.1f}   portfolio equity 10,000 -> {eq:,.0f} "
          f"({(eq/10_000-1)*100:+.1f}%)   maxDD={mdd*100:.1f}%   (risk {args.risk:.1%}/trade)")
    print("-" * 78)
    print("  Read it honestly:")
    print("   * PF > ~1.2 AND positive expectancy that SURVIVES a wider --spread = a real signal.")
    print("   * A smoother curve / lower maxDD than the single-market runs = the diversification working.")
    print("   * PF <= 1.0 or an edge that vanishes at --spread 0.0002 = the basket has no edge either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
