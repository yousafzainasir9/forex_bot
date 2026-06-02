"""Walk-forward backtester — replays history through the LIVE strategy pipeline.

This module reuses the exact same code the live bot uses:
  * bot.indicators.add_indicators  (EMA / RSI / ATR)
  * bot.strategy.evaluate          (BUY / SELL / HOLD / CLOSE)
  * bot.risk.RiskManager           (1% sizing, ATR stop, daily-loss halt, veto)

So a backtest reflects how the bot would actually have behaved — not a separate
re-implementation that could silently disagree.

Honesty / realism choices (a backtest that flatters itself is worthless):
  * No look-ahead — a signal is decided on a CLOSED bar and the trade is entered
    at the NEXT bar's open.
  * Intrabar stop/target — once in a trade, each later bar's high/low is checked
    against SL and TP. If a bar's range spans both, we assume the STOP filled
    first (worst case), never the optimistic outcome.
  * Costs — a configurable spread (crossed once on entry) and per-lot commission
    are deducted from every trade. Default spread is deliberately non-zero.
  * Daily-loss halt — the same 3% rule stops new entries for the rest of a UTC day.

Walk-forward analysis (the part that fights overfitting): the data is split into
consecutive folds; for each fold the best parameters are chosen on an in-sample
slice and then scored on the *next, unseen* out-of-sample slice. Aggregated
out-of-sample results are the honest estimate — in-sample numbers always look good.

A backtest can rule out disasters; it cannot prove a live edge. Treat results as a
sanity check, not a promise.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .risk import DailyState, Decision, RiskManager, SymbolSpec
from .strategy import Action, PositionSide, evaluate


@dataclass
class StrategyParams:
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    atr_multiplier: float = 1.5
    risk_reward: float = 1.5


@dataclass
class CostModel:
    spread_price: float = 0.00012      # ~1.2 pip round of EURUSD; crossed once on entry
    commission_per_lot: float = 0.0    # money per lot, round-turn (e.g. 7.0 on raw accounts)
    slippage_price: float = 0.0        # extra adverse price on entry/exit


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: Dict
    params: Dict

    def render(self) -> str:
        s = self.summary
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        verdict = "PROFIT" if s["net_pnl"] > 0 else ("LOSS" if s["net_pnl"] < 0 else "FLAT")
        return (
            f"Backtest — {verdict} (net {s['net_pnl']:+.2f})\n"
            f"  trades={s['n_trades']} ({s['wins']}W/{s['losses']}L)  "
            f"win%={s['win_rate']:.1%}  PF={pf}\n"
            f"  expectancy/trade={s['expectancy']:+.2f}  avgR={s['avg_r']:+.2f}  "
            f"maxDD={s['max_drawdown']:.2f}\n"
            f"  return={s['return_pct']:+.2f}%  final equity={s['final_equity']:.2f}"
        )


def _summarize(trades: List[dict], starting_equity: float) -> Dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "avg_r": 0.0, "best": 0.0, "worst": 0.0,
                "max_drawdown": 0.0, "final_equity": starting_equity, "return_pct": 0.0}
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    rs = np.array([t["r_multiple"] for t in trades], dtype=float)
    wins = pnls[pnls >= 0]
    losses = pnls[pnls < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum)) if n else 0.0
    final_equity = starting_equity + float(cum[-1])
    return {
        "n_trades": n, "wins": int(len(wins)), "losses": int(len(losses)),
        "win_rate": len(wins) / n,
        "net_pnl": round(float(pnls.sum()), 2),
        "gross_profit": round(gp, 2), "gross_loss": round(gl, 2),
        "profit_factor": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "expectancy": round(float(pnls.mean()), 2),
        "avg_r": round(float(rs.mean()), 3),
        "best": round(float(pnls.max()), 2), "worst": round(float(pnls.min()), 2),
        "max_drawdown": round(max_dd, 2),
        "final_equity": round(final_equity, 2),
        "return_pct": round(100.0 * (final_equity - starting_equity) / starting_equity, 2),
    }


def run_backtest(
    df: pd.DataFrame,
    params: Optional[StrategyParams] = None,
    *,
    starting_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    max_daily_loss: float = 0.03,
    costs: Optional[CostModel] = None,
    symbol_spec: Optional[SymbolSpec] = None,
) -> BacktestResult:
    """Replay ``df`` (OHLC, UTC DatetimeIndex) through the live strategy + risk code."""
    params = params or StrategyParams()
    costs = costs or CostModel()
    spec = symbol_spec or SymbolSpec()

    enriched = add_indicators(
        df, ema_fast=params.ema_fast, ema_slow=params.ema_slow,
        rsi_period=params.rsi_period, atr_period=params.atr_period,
    )
    risk = RiskManager(
        risk_per_trade=risk_per_trade, max_daily_loss=max_daily_loss,
        max_open_positions=1, atr_multiplier=params.atr_multiplier,
        risk_reward=params.risk_reward, symbol_spec=spec,
    )

    n = len(enriched)
    warmup = max(params.ema_slow, params.rsi_period, params.atr_period) + 2
    equity = starting_equity
    pos = None                      # dict: side, entry, sl, tp, lots, risk_amount, open_time
    trades: List[dict] = []
    eq_points = []

    day = None
    day_start_equity = equity
    realized_today = 0.0
    halted = False

    has_time = isinstance(enriched.index, pd.DatetimeIndex)

    def daily_obj() -> DailyState:
        return DailyState(start_equity=day_start_equity, realized_pnl_today=realized_today)

    for i in range(warmup, n):
        bar = enriched.iloc[i]
        bar_time = enriched.index[i]

        # --- UTC day rollover: reset the daily-loss halt ---
        if has_time:
            d = bar_time.date()
            if day is None or d != day:
                day = d
                day_start_equity = equity
                realized_today = 0.0
                halted = False

        # --- 1) manage an open position against THIS bar's range (intrabar) ---
        if pos is not None:
            exit_price = None
            reason = None
            if pos["side"] is PositionSide.LONG:
                if bar["low"] <= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif bar["high"] >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
            else:  # SHORT
                if bar["high"] >= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif bar["low"] <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
            if exit_price is not None:
                pnl = _close_pnl(pos, exit_price, spec, costs)
                equity += pnl
                realized_today += pnl
                trades.append(_trade_row(pos, exit_price, bar_time, pnl, reason))
                eq_points.append((bar_time, equity))
                if daily_obj().loss_fraction() >= max_daily_loss:
                    halted = True
                pos = None

        # --- 2) opposite-cross exit, decided on this CLOSED bar ---
        if pos is not None:
            win = enriched.iloc[max(0, i - 1):i + 1]
            sig = evaluate(win, rsi_overbought=params.rsi_overbought,
                           rsi_oversold=params.rsi_oversold, open_side=pos["side"])
            if sig.action is Action.CLOSE:
                exit_price = float(bar["close"])
                pnl = _close_pnl(pos, exit_price, spec, costs)
                equity += pnl
                realized_today += pnl
                trades.append(_trade_row(pos, exit_price, bar_time, pnl, "opposite-cross"))
                eq_points.append((bar_time, equity))
                if daily_obj().loss_fraction() >= max_daily_loss:
                    halted = True
                pos = None

        # --- 3) entry: decide on closed bar i, fill at bar i+1 open ---
        if pos is None and not halted and i + 1 < n:
            win = enriched.iloc[max(0, i - 1):i + 1]
            sig = evaluate(win, rsi_overbought=params.rsi_overbought,
                           rsi_oversold=params.rsi_oversold, open_side=None)
            if sig.action in (Action.BUY, Action.SELL):
                decision = risk.evaluate(
                    sig, equity=equity, open_positions=0, daily=daily_obj(),
                    kill_switch=False, is_live=False, allow_live=True,
                )
                if decision.approved and decision.plan is not None:
                    plan = decision.plan
                    nxt = enriched.iloc[i + 1]
                    side = plan.side
                    dirn = 1.0 if side is PositionSide.LONG else -1.0
                    # Enter at next open, paying spread + slippage on the adverse side.
                    entry = float(nxt["open"]) + dirn * (costs.spread_price + costs.slippage_price)
                    dist = plan.stop_distance
                    sl = entry - dirn * dist
                    tp = entry + dirn * dist * params.risk_reward
                    pos = {
                        "side": side, "entry": entry, "sl": sl, "tp": tp,
                        "lots": plan.lots, "risk_amount": plan.risk_amount,
                        "open_time": enriched.index[i + 1], "reason_open": plan.reason,
                    }

    # Close any still-open position at the last bar's close (mark-to-market exit).
    if pos is not None:
        last = enriched.iloc[-1]
        pnl = _close_pnl(pos, float(last["close"]), spec, costs)
        equity += pnl
        trades.append(_trade_row(pos, float(last["close"]), enriched.index[-1], pnl, "end-of-data"))
        eq_points.append((enriched.index[-1], equity))

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_points, columns=["time", "equity"]) if eq_points else \
        pd.DataFrame(columns=["time", "equity"])
    summary = _summarize(trades, starting_equity)
    return BacktestResult(trades_df, eq_df, summary, asdict(params))


def _close_pnl(pos: dict, exit_price: float, spec: SymbolSpec, costs: CostModel) -> float:
    dirn = 1.0 if pos["side"] is PositionSide.LONG else -1.0
    raw = (exit_price - pos["entry"]) * dirn * spec.contract_size * pos["lots"]
    # Spread already paid via adverse entry price; here add commission (round-turn).
    commission = costs.commission_per_lot * pos["lots"]
    return round(raw - commission, 2)


def _trade_row(pos: dict, exit_price: float, close_time, pnl: float, reason: str) -> dict:
    r = (pnl / pos["risk_amount"]) if pos["risk_amount"] else 0.0
    return {
        "open_time_utc": str(pos["open_time"]),
        "close_time_utc": str(close_time),
        "side": pos["side"].value,
        "lots": pos["lots"],
        "entry": round(pos["entry"], 6),
        "exit": round(exit_price, 6),
        "stop_loss": round(pos["sl"], 6),
        "take_profit": round(pos["tp"], 6),
        "pnl": pnl,
        "r_multiple": round(r, 3),
        "reason_close": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    folds: List[dict]                 # per-fold chosen params + OOS summary
    oos_trades: pd.DataFrame          # concatenated out-of-sample trades
    oos_summary: Dict                 # aggregate OOS performance (the honest number)
    full_in_sample: Dict              # one in-sample run on all data, for contrast

    def render(self) -> str:
        lines = ["Walk-forward analysis (out-of-sample = the honest estimate)",
                 "=" * 60]
        for f in self.folds:
            p = f["params"]
            s = f["oos"]
            lines.append(
                f"  fold {f['fold']}: EMA {p['ema_fast']}/{p['ema_slow']} "
                f"ATR×{p['atr_multiplier']}  →  OOS net {s['net_pnl']:+.2f} "
                f"({s['n_trades']} trades, PF "
                f"{'inf' if s['profit_factor']==float('inf') else round(s['profit_factor'],2)})"
            )
        s = self.oos_summary
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        lines += ["-" * 60,
                  f"  AGGREGATE OOS: net {s['net_pnl']:+.2f}  trades={s['n_trades']} "
                  f"win%={s['win_rate']:.1%}  PF={pf}  maxDD={s['max_drawdown']:.2f}",
                  f"  (overfit benchmark — best params fitted on ALL data — shows net "
                  f"{self.full_in_sample['net_pnl']:+.2f}; trust the OOS figure above, not this)"]
        return "\n".join(lines)


def default_grid() -> Dict[str, list]:
    """A small, sane search space — kept tiny on purpose to limit overfitting."""
    return {
        "ema_fast": [9, 12],
        "ema_slow": [21, 55],
        "atr_multiplier": [1.5, 2.0],
    }


def _param_combos(grid: Dict[str, list], base: StrategyParams) -> List[StrategyParams]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        kw = asdict(base)
        kw.update(dict(zip(keys, values)))
        if kw["ema_fast"] >= kw["ema_slow"]:
            continue  # invalid: fast must be < slow
        combos.append(StrategyParams(**kw))
    return combos


def _objective(summary: Dict, min_trades: int = 5) -> float:
    """Rank in-sample runs. Penalise too-few-trade fits; reward profit factor."""
    if summary["n_trades"] < min_trades:
        return -1e9
    pf = summary["profit_factor"]
    pf = 5.0 if pf == float("inf") else min(pf, 5.0)  # cap so a 2-trade fluke can't dominate
    return pf * 1000 + summary["net_pnl"]


def walk_forward(
    df: pd.DataFrame,
    grid: Optional[Dict[str, list]] = None,
    *,
    is_bars: int = 2000,
    oos_bars: int = 500,
    base: Optional[StrategyParams] = None,
    **bt_kwargs,
) -> WalkForwardResult:
    """Roll an in-sample/out-of-sample window across the data.

    For each fold: pick the best params on the in-sample slice, then score them on
    the immediately following (unseen) out-of-sample slice. Aggregate the OOS
    trades — that is the estimate that hasn't been fitted to itself.
    """
    grid = grid or default_grid()
    base = base or StrategyParams()
    combos = _param_combos(grid, base)
    n = len(df)
    folds: List[dict] = []
    oos_frames: List[pd.DataFrame] = []
    oos_trades_all: List[dict] = []

    start = 0
    fold_no = 0
    while start + is_bars + oos_bars <= n:
        fold_no += 1
        is_slice = df.iloc[start:start + is_bars]
        oos_slice = df.iloc[start + is_bars:start + is_bars + oos_bars]

        best, best_score = None, -1e18
        for combo in combos:
            res = run_backtest(is_slice, combo, **bt_kwargs)
            score = _objective(res.summary)
            if score > best_score:
                best, best_score = combo, score

        oos_res = run_backtest(oos_slice, best, **bt_kwargs)
        folds.append({"fold": fold_no, "params": asdict(best), "oos": oos_res.summary})
        if not oos_res.trades.empty:
            oos_frames.append(oos_res.trades)
            oos_trades_all.extend(oos_res.trades.to_dict(orient="records"))

        start += oos_bars  # roll forward by one OOS window

    start_eq = bt_kwargs.get("starting_equity", 10_000.0)
    oos_summary = _summarize(oos_trades_all, start_eq)
    oos_trades = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    # The "overfit benchmark": best params fitted on the ENTIRE dataset. This is the
    # flattering, look-ahead-tainted number that walk-forward exists to debunk.
    best_full, best_full_score = None, -1e18
    for combo in combos:
        summ = run_backtest(df, combo, **bt_kwargs).summary
        sc = _objective(summ)
        if sc > best_full_score:
            best_full, best_full_score = summ, sc
    full_is = best_full or run_backtest(df, base, **bt_kwargs).summary
    return WalkForwardResult(folds, oos_trades, oos_summary, full_is)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLC candles from CSV. Needs a time column + open/high/low/close."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("time", "date", "datetime", "timestamp") if c in cols), None)
    if tcol is None:
        raise ValueError("CSV needs a time/date/datetime column.")
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"vol": "volume", "tick_volume": "volume"})
    need = {"open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]


def fetch_mt5_history(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """Pull recent history from MT5 (Windows + terminal only)."""
    from .config import Settings
    from .data import DataFeed
    feed = DataFeed(Settings.load())
    feed.connect()
    try:
        return feed.get_candles(symbol=symbol, timeframe=timeframe, n=bars)
    finally:
        feed.shutdown()


def make_synthetic(n: int = 3000, seed: int = 7, *, freq: str = "5min") -> pd.DataFrame:
    """Deterministic synthetic M5 OHLC for demos/tests (trend + noise + cycles)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    t = np.arange(n)
    drift = 0.00002 * np.sin(t / 180.0)                 # slow regime changes
    steps = rng.normal(0, 0.0004, n) + drift
    close = 1.10 + np.cumsum(steps)
    spread = np.abs(rng.normal(0, 0.0004, n)) + 0.0002
    high = close + spread
    low = close - spread
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close}, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter robustness grid (heatmap) + HTML report
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    df: pd.DataFrame,
    x_param: str,
    x_values: list,
    y_param: str,
    y_values: list,
    *,
    metric: str = "net_pnl",
    base: Optional[StrategyParams] = None,
    **bt_kwargs,
) -> Dict:
    """Run a backtest for every (x, y) parameter pair and collect one metric.

    Returns a matrix you can render as a heatmap. The point isn't to find the single
    best cell — it's to see whether good results form a broad, stable region (robust)
    or a lone spike surrounded by losers (a fluke / overfit). Invalid combos
    (ema_fast >= ema_slow) come back as None.
    """
    base = base or StrategyParams()
    matrix = []
    for yv in y_values:
        row = []
        for xv in x_values:
            kw = asdict(base)
            kw[x_param] = xv
            kw[y_param] = yv
            if kw["ema_fast"] >= kw["ema_slow"]:
                row.append(None)
                continue
            res = run_backtest(df, StrategyParams(**kw), **bt_kwargs)
            val = res.summary.get(metric)
            if val == float("inf"):
                val = None
            row.append(round(val, 2) if isinstance(val, (int, float)) else val)
        matrix.append(row)
    return {"x_param": x_param, "x_values": list(x_values),
            "y_param": y_param, "y_values": list(y_values),
            "metric": metric, "matrix": matrix}


def _heat_color(v: Optional[float], lo: float, hi: float) -> str:
    if v is None:
        return "#1c2230"
    if v >= 0:
        frac = 0.0 if hi <= 0 else min(1.0, v / hi)
        a = 0.12 + 0.55 * frac
        return f"rgba(46,160,67,{a:.2f})"
    frac = 0.0 if lo >= 0 else min(1.0, v / lo)
    a = 0.12 + 0.55 * frac
    return f"rgba(248,81,73,{a:.2f})"


def _heatmap_html(hm: Optional[Dict]) -> str:
    if not hm:
        return ""
    vals = [v for row in hm["matrix"] for v in row if v is not None]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 0.0
    head = "<th></th>" + "".join(f"<th>{hm['x_param']}={x}</th>" for x in hm["x_values"])
    rows = ""
    for yv, row in zip(hm["y_values"], hm["matrix"]):
        cells = f"<th>{hm['y_param']}={yv}</th>"
        for v in row:
            txt = "—" if v is None else f"{v:+.0f}"
            cells += f'<td style="background:{_heat_color(v, lo, hi)}">{txt}</td>'
        rows += f"<tr>{cells}</tr>"
    return (
        f'<div class="panel"><h3>Robustness heatmap — {hm["metric"]} '
        f'({hm["y_param"]} × {hm["x_param"]})</h3>'
        f'<table class="heat"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
        '<p class="small">Look for a broad coloured region, not one lone green cell — '
        'a single isolated winner usually means an overfit fluke.</p></div>'
    )


def build_backtest_html(result: BacktestResult, heatmap: Optional[Dict] = None,
                        *, title: str = "Backtest report") -> str:
    """Render a self-contained HTML report: summary, equity curve, heatmap, trades."""
    import json as _json
    s = result.summary
    pf = "&#8734;" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    verdict = "PROFIT" if s["net_pnl"] > 0 else ("LOSS" if s["net_pnl"] < 0 else "FLAT")
    vcls = "pos" if s["net_pnl"] > 0 else ("neg" if s["net_pnl"] < 0 else "")
    eq = result.equity_curve
    eq_labels = [str(t)[:16].replace("T", " ") for t in eq["time"]] if not eq.empty else []
    eq_data = [round(float(v), 2) for v in eq["equity"]] if not eq.empty else []

    def card(label, val, cls=""):
        return (f'<div class="card"><div class="label">{label}</div>'
                f'<div class="val {cls}">{val}</div></div>')

    cards = "".join([
        card("Result", verdict, vcls),
        card("Net P&amp;L", f'{s["net_pnl"]:+.2f}', vcls),
        card("Return", f'{s["return_pct"]:+.2f}%', vcls),
        card("Trades", f'{s["n_trades"]} <span class="small">({s["wins"]}W/{s["losses"]}L)</span>'),
        card("Win rate", f'{s["win_rate"]*100:.1f}%'),
        card("Profit factor", pf),
        card("Avg R", f'{s["avg_r"]:+.2f}'),
        card("Max drawdown", f'{s["max_drawdown"]:.2f}'),
    ])

    trows = ""
    if not result.trades.empty:
        for _, t in result.trades.iterrows():
            pcls = "pos" if t["pnl"] >= 0 else "neg"
            trows += (f'<tr><td>{str(t["close_time_utc"])[:16].replace("T"," ")}</td>'
                      f'<td>{t["side"]}</td><td>{t["lots"]}</td><td>{t["entry"]}</td>'
                      f'<td>{t["exit"]}</td><td class="{pcls}">{t["pnl"]:+.2f}</td>'
                      f'<td class="{pcls}">{t["r_multiple"]:+.2f}</td>'
                      f'<td>{t["reason_close"]}</td></tr>')
    trades_tbl = (f'<div class="panel"><h3>Trades ({len(result.trades)})</h3>'
                  f'<table><thead><tr><th>Close (UTC)</th><th>Side</th><th>Lots</th>'
                  f'<th>Entry</th><th>Exit</th><th>Net P&amp;L</th><th>R</th><th>Reason</th>'
                  f'</tr></thead><tbody>{trows}</tbody></table></div>') if trows else \
                 '<div class="panel"><div class="empty">No trades.</div></div>'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
 :root{{--bg:#0e1117;--panel:#161b22;--line:#2a3140;--txt:#e6edf3;--muted:#8b949e;--green:#2ea043;--red:#f85149}}
 body{{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
 h1{{font-size:18px;padding:18px 24px;margin:0;border-bottom:1px solid var(--line)}}
 .wrap{{padding:18px 24px;max-width:1100px;margin:0 auto}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:18px}}
 .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}}
 .card .label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
 .card .val{{font-size:21px;font-weight:700;margin-top:6px}}
 .pos{{color:var(--green)}} .neg{{color:var(--red)}}
 .panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:18px}}
 .panel h3{{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--muted)}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}}
 th{{color:var(--muted);font-weight:600}}
 table.heat td{{text-align:center;border:1px solid var(--bg);font-weight:600}}
 table.heat th{{text-align:center}}
 .small{{font-size:11px;color:var(--muted)}} .empty{{color:var(--muted);text-align:center;padding:24px}}
</style></head><body>
<h1>{title} — {verdict} ({s['net_pnl']:+.2f})</h1>
<div class="wrap">
 <div class="cards">{cards}</div>
 <div class="panel"><h3>Equity curve (account value after each closed trade)</h3>
   <canvas id="eq" height="200"></canvas></div>
 {_heatmap_html(heatmap)}
 {trades_tbl}
</div>
<script>
 const L={_json.dumps(eq_labels)}, D={_json.dumps(eq_data)};
 if(window.Chart && D.length){{new Chart(document.getElementById("eq"),{{type:"line",
   data:{{labels:L,datasets:[{{data:D,borderColor:"#58a6ff",backgroundColor:"rgba(88,166,255,.12)",
   fill:true,tension:.2,pointRadius:0,borderWidth:2}}]}},
   options:{{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:"#8b949e",maxTicksLimit:8}},grid:{{color:"#2a3140"}}}},
   y:{{ticks:{{color:"#8b949e"}},grid:{{color:"#2a3140"}}}}}}}}}});}}
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _params_from_settings() -> StrategyParams:
    try:
        from .config import Settings
        s = Settings.load()
        return StrategyParams(
            ema_fast=s.ema_fast, ema_slow=s.ema_slow, rsi_period=s.rsi_period,
            rsi_overbought=s.rsi_overbought, rsi_oversold=s.rsi_oversold,
            atr_period=s.atr_period, atr_multiplier=s.atr_multiplier,
            risk_reward=s.risk_reward,
        )
    except Exception:
        return StrategyParams()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward backtester for the M5 bot.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="OHLC CSV (time,open,high,low,close).")
    src.add_argument("--demo", action="store_true", help="use built-in synthetic data.")
    src.add_argument("--symbol", help="pull history from MT5 for this symbol (Windows).")
    p.add_argument("--timeframe", default="M5")
    p.add_argument("--bars", type=int, default=5000, help="bars to pull from MT5.")
    p.add_argument("--walk-forward", action="store_true", help="run walk-forward analysis.")
    p.add_argument("--is-bars", type=int, default=2000, help="in-sample window (walk-forward).")
    p.add_argument("--oos-bars", type=int, default=500, help="out-of-sample window (walk-forward).")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--spread", type=float, default=0.00012, help="spread in price (e.g. 0.00012 = 1.2 pips).")
    p.add_argument("--commission", type=float, default=0.0, help="commission per lot (round-turn).")
    p.add_argument("--report", action="store_true", help="write a self-contained HTML report (single run).")
    p.add_argument("--metric", default="net_pnl", help="heatmap metric (net_pnl/profit_factor/return_pct/avg_r).")
    args = p.parse_args(argv)

    if args.csv:
        df = load_csv(args.csv)
    elif args.symbol:
        df = fetch_mt5_history(args.symbol, args.timeframe, args.bars)
    else:
        df = make_synthetic()
        if not args.demo:
            print("(no --csv/--symbol given; using --demo synthetic data)")

    costs = CostModel(spread_price=args.spread, commission_per_lot=args.commission)
    bt_kwargs = dict(starting_equity=args.equity, risk_per_trade=args.risk, costs=costs)
    out_dir = Path(__file__).resolve().parent.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.walk_forward:
        wf = walk_forward(df, is_bars=args.is_bars, oos_bars=args.oos_bars,
                          base=_params_from_settings(), **bt_kwargs)
        print(wf.render())
        if not wf.oos_trades.empty:
            wf.oos_trades.to_csv(out_dir / "backtest_oos_trades.csv", index=False)
            print(f"\nOOS trades → {out_dir / 'backtest_oos_trades.csv'}")
    else:
        base = _params_from_settings()
        res = run_backtest(df, base, **bt_kwargs)
        print(res.render())
        if not res.trades.empty:
            res.trades.to_csv(out_dir / "backtest_trades.csv", index=False)
            print(f"\nTrades -> {out_dir / 'backtest_trades.csv'}")
        if args.report:
            hm = grid_search(
                df, "ema_fast", [5, 8, 9, 12], "ema_slow", [21, 34, 55],
                metric=args.metric, base=base, **bt_kwargs,
            )
            html = build_backtest_html(res, hm, title=f"Backtest — {args.metric}")
            (out_dir / "backtest_report.html").write_text(html, encoding="utf-8")
            print(f"Report -> {out_dir / 'backtest_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
