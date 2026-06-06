"""Option 2 — ML trade-quality FILTER for the basket strategy.

IDEA: keep the entry signal exactly as-is, but LEARN TO RANK the signals it
already produces and only take the ones a model predicts are worth taking. This
does NOT try to predict price with AI (that's the hype trap) — it predicts
"given that a signal just fired, is THIS one likely to be a winner?" using the
context a human would glance at.

HOW IT WORKS (and why it's honest):
  1. Run the existing backtest across the basket -> every trade the strategy took,
     with its realised R-multiple (the LABEL: win if R > 0).
  2. For each trade, attach a feature vector computed CAUSALLY at the SIGNAL bar
     (the closed bar the decision was made on) — indicators, recent price action,
     volatility, time-of-day, direction. Only information available before entry.
  3. Walk forward in TIME: train a gradient-boosted classifier on PAST trades,
     score FUTURE (unseen) trades, keep only the ones it rates above average, and
     compare that subset's expectancy / profit-factor to taking ALL trades.

LEAKAGE GUARDRAILS (this is what makes or breaks ML in trading):
  * Features use only data up to the signal bar (causal — no peeking ahead).
  * Train/test is strictly TEMPORAL (never shuffled): train on the past, test on
    the future, exactly like the rest of our walk-forward.
  * The model only RE-RANKS trades the strategy already fired. If the features
    carry no information about outcomes, no threshold helps — and the report says
    so plainly instead of flattering itself.

Needs scikit-learn:   uv add scikit-learn      (or: pip install scikit-learn)

Run (Windows, MT5 up & logged into the demo):
    uv run python ml_filter.py --timeframe H1 --htf H4 --bars 20000
    uv run python ml_filter.py --timeframe H1 --htf H4 --bars 20000 --spread 0.0002
    uv run python ml_filter.py --timeframe M5 --htf H1 --bars 60000   (more samples)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bot.backtest import run_backtest, fetch_mt5_history, _params_from_settings, CostModel
from bot.indicators import add_indicators

DEFAULT_BASKET = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
                  "USDCAD", "NZDUSD", "XAUUSD"]

# Features fed to the model. Directional ones are signed by trade direction so
# "+" always means "in the trade's favour", letting one model serve longs+shorts.
FEATURES = ["ema_gap_atr", "rsi_dir", "adx", "atr_pct", "ret5", "ret20",
            "dist_hi", "dist_lo", "hour", "dow", "side"]


def _feature_frame(df: pd.DataFrame, params) -> pd.DataFrame:
    """Indicator-enriched frame plus engineered, CAUSAL context features."""
    e = add_indicators(df, ema_fast=params.ema_fast, ema_slow=params.ema_slow,
                       rsi_period=params.rsi_period, atr_period=params.atr_period,
                       adx_period=params.adx_period).copy()
    atr = e["atr"].replace(0.0, np.nan)
    e["ema_gap_atr"] = (e["ema_fast"] - e["ema_slow"]) / atr
    e["atr_pct"] = e["atr"] / e["close"]
    e["ret5"] = e["close"].pct_change(5)
    e["ret20"] = e["close"].pct_change(20)
    e["dist_hi"] = (e["high"].rolling(20).max() - e["close"]) / atr
    e["dist_lo"] = (e["close"] - e["low"].rolling(20).min()) / atr
    if isinstance(e.index, pd.DatetimeIndex):
        e["hour"] = e.index.hour
        e["dow"] = e.index.dayofweek
    else:
        e["hour"] = 0
        e["dow"] = 0
    return e


def collect(symbols, tf, bars, params, bt) -> pd.DataFrame:
    """Run the backtest per symbol; pool (features @ signal bar, R outcome) rows."""
    rows = []
    for sym in symbols:
        try:
            df = fetch_mt5_history(sym, tf, bars)
        except Exception as e:
            print(f"  {sym:8} skip — no data ({e})")
            continue
        res = run_backtest(df, params, **bt)
        if res.trades.empty:
            print(f"  {sym:8} 0 trades")
            continue
        feat = _feature_frame(df, params)
        pos_of = {t: k for k, t in enumerate(feat.index)}
        n = 0
        for _, tr in res.trades.iterrows():
            ot = pd.to_datetime(tr["open_time_utc"], utc=True, errors="coerce")
            k = pos_of.get(ot)
            if k is None:
                continue
            sig = max(0, k - 1)            # decision bar = the bar BEFORE the fill
            row = feat.iloc[sig]
            side = 1 if tr["side"] == "LONG" else -1

            def g(name, default=0.0):
                v = row.get(name, default)
                return float(v) if v == v else default   # NaN -> default

            rec = {
                "time": feat.index[sig], "symbol": sym,
                "r": float(tr["r_multiple"]),
                "win": 1 if float(tr["r_multiple"]) > 0 else 0,
                "ema_gap_atr": g("ema_gap_atr") * side,
                "rsi_dir": (g("rsi", 50.0) - 50.0) * side,     # momentum in trade direction
                "adx": g("adx"),
                "atr_pct": g("atr_pct"),
                "ret5": g("ret5") * side,
                "ret20": g("ret20") * side,
                "dist_hi": g("dist_hi"),
                "dist_lo": g("dist_lo"),
                "hour": int(g("hour")), "dow": int(g("dow")), "side": side,
            }
            if all(np.isfinite(rec[f]) for f in FEATURES):
                rows.append(rec)
                n += 1
        print(f"  {sym:8} {n} labeled trades")
    return pd.DataFrame(rows)


def evaluate(ds: pd.DataFrame, n_splits: int = 5):
    """Temporal walk-forward: train on the past, score the future, keep the trades
    the model rates above the training base-rate. Returns (all_R, kept_R)."""
    from sklearn.ensemble import GradientBoostingClassifier

    ds = ds.sort_values("time").reset_index(drop=True)
    n = len(ds)
    step = n // (n_splits + 1)
    all_r, kept_r = [], []
    if step < 20:
        return np.array([]), np.array([])
    for s in range(1, n_splits + 1):
        tr = ds.iloc[: step * s]
        te = ds.iloc[step * s: step * (s + 1)]
        if len(te) == 0 or tr["win"].nunique() < 2:
            continue
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=0)
        clf.fit(tr[FEATURES], tr["win"])
        thr = float(tr["win"].mean())     # take setups scored above the base win-rate
        p = clf.predict_proba(te[FEATURES])[:, 1]
        all_r.extend(te["r"].tolist())
        kept_r.extend(te["r"][p >= thr].tolist())
    return np.array(all_r), np.array(kept_r)


def _stats(r):
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    pf = (gp / gl) if gl > 0 else float("inf")
    return len(r), 100.0 * np.mean(r > 0), pf, float(np.mean(r))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ML trade-quality filter (Option 2).")
    p.add_argument("--symbols", default=",".join(DEFAULT_BASKET))
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--htf", default="H4")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--spread", type=float, default=0.00012)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--no-htf-gate", action="store_true")
    args = p.parse_args(argv)

    from bot.config import Settings
    s = Settings.load()
    costs = CostModel(spread_price=args.spread)
    bt = dict(
        starting_equity=10_000.0, risk_per_trade=args.risk, costs=costs,
        ride_trend_after_tp=s.ride_trend_after_tp, risk_mode="fixed",
        require_htf_align=(not args.no_htf_gate and s.require_htf_align),
        htf_timeframe=args.htf, session_filter=s.session_filter,
        session_start_hour=s.session_start_hour, session_end_hour=s.session_end_hour,
        trail_after_tp=s.trail_after_tp, trail_atr_mult=s.trail_atr_mult,
        partial_tp_enabled=s.partial_tp_enabled, partial_tp_fraction=s.partial_tp_fraction,
        partial_tp_r=s.partial_tp_r, lock_profit_r=s.lock_profit_r,
        ride_stall_atr_frac=s.ride_stall_atr_frac, max_bars_in_trade=s.max_bars_in_trade,
    )
    params = _params_from_settings()
    symbols = [x.strip().upper()
               for c in args.symbols.split(",") for x in c.split() if x.strip()]

    print(f"Collecting labeled trades | {args.timeframe} (HTF {args.htf}) | "
          f"{len(symbols)} symbols | spread {args.spread}")
    print("-" * 64)
    ds = collect(symbols, args.timeframe, args.bars, params, bt)
    print("-" * 64)
    print(f"Total labeled trades: {len(ds)}")
    if len(ds) < 300:
        print("WARNING: <300 samples is far too few to train an ML filter without")
        print("overfitting — use M5 and/or more symbols/bars before trusting this.")
    if len(ds) == 0:
        return 0

    all_r, kept_r = evaluate(ds)
    if len(all_r) == 0:
        print("Not enough data for a temporal walk-forward. Increase --bars or symbols.")
        return 0

    print("\n--- OOS comparison (temporal walk-forward, train-past / test-future) ---")
    n, w, pf, e = _stats(all_r)
    print(f"  ALL signals  : trades={n:4d}  win%={w:5.1f}  PF={pf:5.2f}  exp={e:+.3f} R/trade")
    n, w, pf, e = _stats(kept_r)
    print(f"  ML-FILTERED  : trades={n:4d}  win%={w:5.1f}  PF={pf:5.2f}  exp={e:+.3f} R/trade")
    print("-" * 64)
    print("  Read it honestly:")
    print("   * FILTERED PF & expectancy clearly ABOVE 'ALL' = the model found real")
    print("     trade-quality information; worth integrating + cost-stressing + demo.")
    print("   * Similar or worse (or far fewer trades for no gain) = the features")
    print("     carry no edge. No model fixes that — don't deploy it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
