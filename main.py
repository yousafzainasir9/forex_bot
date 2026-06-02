"""Runner — orchestrates the full pipeline once per CLOSED candle.

Pipeline:  Config -> Data -> Indicators -> Strategy -> [Scanner ranks watchlist]
           -> Risk -> Execution -> Monitor

The bot now watches a WHOLE WATCHLIST (settings.symbols), not just one pair. Each
cycle it scores every symbol's opportunity and acts according to settings.scan_mode:
  * best  — trade only the single strongest opportunity at a time (1 position).
  * top_n — keep up to SCAN_TOP_N positions, opening the best still-flat symbols.
  * any   — open every qualifying symbol up to MAX_OPEN_POSITIONS.

Usage
-----
  uv run python main.py --once         # single dry pass, LOG ONLY (no orders)
  uv run python main.py                # full demo loop
  uv run python main.py --symbol GBPUSD  # watch only one symbol this run

Web app integration
-------------------
  * logs/status.json   — live snapshot (now includes the active symbol).
  * logs/candles.json  — candles + EMAs for the ACTIVE symbol (charted).
  * logs/scan.json     — the ranked watchlist the dashboard renders.
  * logs/control.json  — kill switch the web app flips live.

RISK WARNING: educational software for DEMO accounts. Most retail forex bots lose
money. Never run against real funds until you fully understand this code.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from bot.config import Settings
from bot.control import kill_active
from bot.data import DataFeed
from bot.execution import Executor
from bot.indicators import add_indicators
from bot.monitor import Monitor
from bot.risk import DailyState, RiskManager
from bot.scanner import scan_watchlist, rank_opportunities
from bot.strategy import Action, PositionSide, evaluate, trend_ride_exit


_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


class Bot:
    def __init__(self, settings: Settings, *, dry_run: bool = False) -> None:
        self.s = settings
        self.dry_run = dry_run
        self.mon = Monitor(settings.log_dir)
        self.feed = DataFeed(settings)
        self.execu = Executor(settings, monitor=self.mon)
        self.risk = RiskManager(
            risk_per_trade=settings.risk_per_trade,
            max_risk_per_trade=settings.max_risk_per_trade,
            max_daily_loss=settings.max_daily_loss,
            risk_mode=settings.risk_mode,
            max_open_positions=settings.max_open_positions,
            atr_multiplier=settings.atr_multiplier,
            risk_reward=settings.risk_reward,
        )
        self.daily: Optional[DailyState] = None
        self.active_symbol = settings.symbol
        self._last_signal = None
        self._last_scan = []
        self._traded_bar = {}   # symbol -> bar_time we already acted on
        self._spec_cache = {}   # symbol -> SymbolSpec
        self._stop = False

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.s.assert_safe_to_run()
        self.mon.info("=" * 70)
        self.mon.info(f"Forex bot starting — {self.s.banner()}")
        if self.dry_run:
            self.mon.info("DRY RUN (--once): log only, NO orders will be placed.")
        acct = self.feed.connect()
        self.mon.info(f"Connected: login={acct.login} {acct.trade_mode} "
                      f"equity={acct.equity} {acct.currency} @ {acct.server}")
        # Probe the watchlist; drop symbols the broker doesn't offer.
        available = []
        for sym in self.s.symbols:
            if self.feed.ensure_symbol(sym):
                available.append(sym)
            else:
                self.mon.warning(f"Watchlist: {sym} not offered by broker — skipping.")
        if available:
            import dataclasses
            self.s = dataclasses.replace(self.s, symbols=available)
            if self.active_symbol not in available:
                self.active_symbol = available[0]
        self.mon.info(f"Watchlist active ({len(self.s.symbols)}): {', '.join(self.s.symbols)}")
        self.daily = DailyState(start_equity=acct.equity)
        self.mon.maybe_rollover(equity=acct.equity)
        self._startup_backfill()
        self._write_status(acct)

    def _startup_backfill(self) -> None:
        """One-time backfill: pull closed trades from the last
        ``reconcile_lookback_days`` into the ledger. The per-loop reconcile only
        scans the current UTC day, so without this a trade that closed earlier (or
        while the bot was off) would never reach trades.csv. The bot owns a stable
        MT5 connection, so this is more reliable than the web app's best-effort
        reconcile (which can lose the IPC race with the bot). Idempotent (deduped
        by position id); does NOT touch today's daily-loss accounting."""
        if self.dry_run:
            return
        try:
            days = max(1, int(self.s.reconcile_lookback_days))
            since = datetime.now(timezone.utc) - timedelta(days=days)
            before = self.mon.stats.n_trades
            self.execu.reconcile_closed_trades(self.mon, since)
            # Then directly clean up any position the bot still tracks as "open"
            # but that has actually closed at the broker — the window reconcile can
            # miss broker TP/SL closes. Record each by ticket (magic-agnostic).
            try:
                tracked = list(self.mon._read_open_map().keys())
                open_tickets = {str(p.ticket) for p in self.execu.open_positions_all()}
                for tkt in tracked:
                    if tkt not in open_tickets:
                        self.execu.record_closed_position(self.mon, int(tkt))
            except Exception as e:
                self.mon.warning(f"Open-trade cleanup skipped: {e}")
            added = self.mon.stats.n_trades - before
            self.mon.info(
                f"Startup backfill: scanned last {days} day(s), "
                f"recorded {added} previously-unlogged closed trade(s).")
        except Exception as e:
            self.mon.warning(f"Startup backfill skipped: {e}")

    def shutdown(self) -> None:
        try:
            acct = None
            try:
                acct = self.feed.account_info()
            except Exception:
                pass
            try:
                self._reconcile(equity=acct.equity if acct else None)
            except Exception as e:
                self.mon.warning(f"Final reconcile skipped: {e}")
            self.mon.daily_summary(equity=acct.equity if acct else None)
            try:
                from bot.dashboard import build_html
                (self.s.log_dir / "dashboard.html").write_text(
                    build_html(self.mon.trades_csv), encoding="utf-8")
            except Exception as e:
                self.mon.warning(f"Dashboard refresh skipped: {e}")
            try:
                self._write_status(acct, running=False)
            except Exception:
                pass
        finally:
            self.feed.shutdown()
            self.mon.info(f"Shutdown complete. {self.mon.stats_line()}")

    def _kill(self) -> bool:
        return bool(self.s.kill_switch) or kill_active(self.s.log_dir)

    def _spec_for(self, symbol: str):
        if symbol not in self._spec_cache:
            try:
                self._spec_cache[symbol] = self.feed.symbol_spec(symbol)
            except Exception as e:
                self.mon.warning(f"symbol_spec({symbol}) failed, using FX defaults: {e}")
                from bot.risk import SymbolSpec
                self._spec_cache[symbol] = SymbolSpec()
        return self._spec_cache[symbol]

    # --- live status snapshot for the web app ----------------------------
    def _write_status(self, acct=None, *, running: bool = True) -> None:
        try:
            positions = [] if self.dry_run else self.execu.open_positions_all()
            pos_out = [{
                "ticket": p.ticket, "symbol": p.symbol, "side": p.side.value,
                "lots": p.lots, "entry": p.entry, "stop_loss": p.stop_loss,
                "take_profit": p.take_profit, "profit": p.profit,
            } for p in positions]
            sig = self._last_signal
            s = self.stats = self.mon.stats
            status = {
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "running": running,
                "mode": "LIVE" if self.s.is_live else "DEMO",
                "symbol": self.active_symbol,
                "primary_symbol": self.s.symbol,
                "watchlist_size": len(self.s.symbols),
                "scan_mode": self.s.scan_mode,
                "timeframe": self.s.timeframe,
                "kill_switch": self._kill(),
                "equity": getattr(acct, "equity", None),
                "balance": getattr(acct, "balance", None),
                "currency": getattr(acct, "currency", None),
                "open_positions": pos_out,
                "last_signal": None if sig is None else {
                    "action": sig.action.value, "reason": sig.reason,
                    "price": sig.price, "rsi": sig.rsi,
                    "symbol": self.active_symbol,
                    "time": sig.bar_time.isoformat() if sig.bar_time is not None else None,
                },
                "daily_realized_pnl": round(self.daily.realized_pnl_today, 2) if self.daily else 0.0,
                "daily_loss_fraction": round(self.daily.loss_fraction(), 4) if self.daily else 0.0,
                "max_daily_loss": self.s.max_daily_loss,
                "risk": (lambda e: (lambda r: {
                    "mode": self.s.risk_mode,
                    "per_trade": round(r["risk_per_trade"], 4),
                    "max_per_trade": round(r["max_risk_per_trade"], 4),
                    "daily_stop": round(r["max_daily_loss"], 4),
                })(self.risk.effective_risk(e)))(getattr(acct, "equity", 0.0) or 0.0),
                "stats": {
                    "n_trades": s.n_trades, "wins": s.wins, "losses": s.losses,
                    "win_rate": round(s.win_rate, 4),
                    "net_pnl": round(s.net_pnl, 2),
                    "profit_factor": (None if s.profit_factor == float("inf") else round(s.profit_factor, 2)),
                    "max_drawdown": round(s.max_drawdown, 2),
                },
            }
            (self.s.log_dir / "status.json").write_text(
                json.dumps(status, indent=2), encoding="utf-8")
        except Exception as e:
            self.mon.warning(f"status.json write skipped: {e}")

    # --- ranked-watchlist snapshot for the web app -----------------------
    def _write_scan(self, scans, ranked) -> None:
        try:
            rows = [sc.to_dict() for sc in sorted(
                scans, key=lambda x: (x.tradable, x.score), reverse=True)]
            payload = {
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "timeframe": self.s.timeframe,
                "htf_timeframe": self.s.htf_timeframe,
                "scan_mode": self.s.scan_mode,
                "active_symbol": self.active_symbol,
                "top": ranked[0].symbol if ranked else None,
                "rows": rows,
            }
            (self.s.log_dir / "scan.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception as e:
            self.mon.warning(f"scan.json write skipped: {e}")

    # --- live candle snapshot for the active symbol ----------------------
    def _write_candles(self, symbol: str, n: int = 180) -> None:
        try:
            candles = self.feed.get_candles(symbol=symbol, timeframe=self.s.timeframe,
                                            n=self.s.history_bars)
            enriched = add_indicators(candles, ema_fast=self.s.ema_fast,
                                      ema_slow=self.s.ema_slow,
                                      rsi_period=self.s.rsi_period,
                                      atr_period=self.s.atr_period)
            df = enriched.tail(n)
            out, ef, es = [], [], []
            for ts, row in df.iterrows():
                t = int(ts.timestamp())
                out.append({"t": t, "o": round(float(row["open"]), 6),
                            "h": round(float(row["high"]), 6),
                            "l": round(float(row["low"]), 6),
                            "c": round(float(row["close"]), 6)})
                if not pd.isna(row.get("ema_fast")):
                    ef.append({"t": t, "v": round(float(row["ema_fast"]), 6)})
                if not pd.isna(row.get("ema_slow")):
                    es.append({"t": t, "v": round(float(row["ema_slow"]), 6)})
            payload = {"symbol": symbol, "timeframe": self.s.timeframe,
                       "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "candles": out, "ema_fast": ef, "ema_slow": es}
            (self.s.log_dir / "candles.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception as e:
            self.mon.warning(f"candles.json write skipped: {e}")

    # --- order helper ----------------------------------------------------
    def _try_open(self, scan, equity: float, open_count: int, kill: bool) -> bool:
        """Risk-evaluate and (if approved) open a position for one scanned symbol."""
        sym = scan.symbol
        self.risk.spec = self._spec_for(sym)
        sig = evaluate(
            add_indicators(
                self.feed.get_candles(symbol=sym, timeframe=self.s.timeframe, n=self.s.history_bars),
                ema_fast=self.s.ema_fast, ema_slow=self.s.ema_slow,
                rsi_period=self.s.rsi_period, atr_period=self.s.atr_period),
            rsi_overbought=self.s.rsi_overbought, rsi_oversold=self.s.rsi_oversold,
            open_side=None,
        )
        if sig.action not in (Action.BUY, Action.SELL):
            return False
        # one entry per symbol per bar
        if self._traded_bar.get(sym) == sig.bar_time:
            return False
        decision = self.risk.evaluate(
            sig, equity=equity, open_positions=open_count,
            daily=self.daily or DailyState(equity),
            kill_switch=kill, is_live=self.s.is_live, allow_live=self.s.is_live,
        )
        self.mon.log_risk(decision)
        if not (decision.approved and decision.plan is not None):
            return False
        res = self.execu.open_position(decision.plan, symbol=sym)
        if res.ok and res.ticket is not None:
            self._traded_bar[sym] = sig.bar_time
            pid = self.execu.find_position_id(res.ticket, symbol=sym)
            plan = decision.plan
            self.mon.note_open(pid, {
                "symbol": sym, "side": plan.side.value,
                "stop_loss": plan.stop_loss, "take_profit": plan.take_profit,
                "risk_amount": plan.risk_amount, "reason_open": plan.reason,
            })
            return True
        return False

    # --- one pipeline pass ----------------------------------------------
    def run_pipeline(self) -> None:
        positions = [] if self.dry_run else self.execu.open_positions_all()
        open_map = {p.symbol: p.side for p in positions}

        scans = scan_watchlist(self.feed, self.s, open_symbols=open_map)
        ranked = rank_opportunities(scans)
        self._last_scan = scans

        # Choose the active (charted) symbol: an open position wins, else the top
        # opportunity, else the primary symbol.
        if positions:
            self.active_symbol = positions[0].symbol
        elif ranked:
            self.active_symbol = ranked[0].symbol
        else:
            self.active_symbol = self.s.symbol

        # Remember the active symbol's signal for the status card.
        active_scan = next((s for s in scans if s.symbol == self.active_symbol), None)
        if active_scan is not None:
            top_txt = (f"{ranked[0].symbol} ({ranked[0].score:.3f})" if ranked else "none")
            self.mon.info(
                f"SCAN active={self.active_symbol} {active_scan.action} "
                f"score={active_scan.score:.3f} | top={top_txt} "
                f"| tradable={len(ranked)}/{len(scans)}")

        self._write_scan(scans, ranked)
        self._write_candles(self.active_symbol)

        kill = self._kill()

        if self.dry_run:
            acct = self.feed.account_info()
            top = ranked[0] if ranked else None
            if top is not None:
                self.mon.info(f"DRY top opportunity: {top.symbol} {top.action} "
                              f"score={top.score:.3f} ({top.skip_reason or 'tradable'})")
            return

        # --- 1) Manage existing positions: close on opposite cross, else (when
        #        trend-riding) ride past the target and exit on a turning bar. ---
        for p in positions:
            sc = next((s for s in scans if s.symbol == p.symbol), None)
            if sc is not None and sc.action == Action.CLOSE.value:
                self.mon.info(f"Exit signal on {p.symbol}: {sc.reason}")
                self.execu.close_position(p.ticket)
                continue
            if self.s.ride_trend_after_tp:
                self._trend_ride_check(p)

        # --- 2) Open new positions per scan_mode. ---
        if kill or not ranked:
            return
        acct = self.feed.account_info()
        equity = acct.equity
        open_now = self.execu.open_positions_all()
        open_syms = {p.symbol for p in open_now}

        if self.s.scan_mode == "best":
            # Only act when completely flat; trade the single strongest.
            if not open_now:
                top = ranked[0]
                if top.score >= self.s.scan_min_score:
                    self._last_signal = self._signal_for(top.symbol)
                    self._try_open(top, equity, open_count=0, kill=kill)
        else:
            cap = (self.s.scan_top_n if self.s.scan_mode == "top_n"
                   else self.s.max_open_positions)
            for cand in ranked:
                if len(open_syms) >= cap:
                    break
                if cand.symbol in open_syms:
                    continue
                if cand.score < self.s.scan_min_score:
                    continue
                if self._try_open(cand, equity, open_count=len(open_syms), kill=kill):
                    open_syms.add(cand.symbol)
                    self._last_signal = self._signal_for(cand.symbol)

    def _open_entry(self, p):
        """Find this open position's bookkeeping row (key, info, full map).

        Keyed by position id at open time, which normally equals the position
        ticket; falls back to a symbol+side match so the target is still found if
        the broker reports a different identifier.
        """
        m = self.mon._read_open_map()
        k = str(p.ticket)
        if k in m:
            return k, m[k], m
        for kk, v in m.items():
            if v.get("symbol") == p.symbol and v.get("side") == p.side.value:
                return kk, v, m
        return None, {}, m

    def _trend_ride_check(self, p) -> None:
        """Once a position reaches its target, ride the trend: hold while each new
        closed bar keeps printing higher (LONG) / lower (SHORT) closes, and exit
        only when a bar closes against the trade. The broker stop-loss still caps
        the downside throughout."""
        key, info, _ = self._open_entry(p)
        tp = float(info.get("take_profit", 0) or 0)
        if tp <= 0:
            return  # no stored target — nothing to ride (e.g. externally opened)
        try:
            candles = self.feed.get_candles(symbol=p.symbol, timeframe=self.s.timeframe, n=10)
        except Exception as e:
            self.mon.warning(f"trend-ride: candle fetch failed for {p.symbol}: {e}")
            return
        if candles is None or len(candles) < 2:
            return
        last, prev = candles.iloc[-1], candles.iloc[-2]
        dec = trend_ride_exit(
            side=p.side, take_profit=tp,
            prev_high=float(prev["high"]), prev_low=float(prev["low"]),
            last_high=float(last["high"]), last_low=float(last["low"]),
            tp_reached=bool(info.get("tp_reached", False)),
        )
        # Latch the TP-reached state into the bookkeeping file so it survives loops.
        if dec.tp_reached and not info.get("tp_reached") and key is not None:
            info["tp_reached"] = True
            m = self.mon._read_open_map()
            m[key] = {**m.get(key, {}), **info}
            self.mon._write_open_map(m)
            self.mon.info(f"{p.symbol}: target {tp} reached — trend-riding "
                          f"(won't close until a turning {self.s.timeframe} bar).")
        if dec.should_close:
            self.mon.info(f"{p.symbol}: {dec.reason} — closing.")
            self.execu.close_position(p.ticket)

    def _signal_for(self, symbol: str):
        """Best-effort recompute of the strategy signal for status display."""
        try:
            enr = add_indicators(
                self.feed.get_candles(symbol=symbol, timeframe=self.s.timeframe, n=self.s.history_bars),
                ema_fast=self.s.ema_fast, ema_slow=self.s.ema_slow,
                rsi_period=self.s.rsi_period, atr_period=self.s.atr_period)
            return evaluate(enr, rsi_overbought=self.s.rsi_overbought,
                            rsi_oversold=self.s.rsi_oversold)
        except Exception:
            return self._last_signal

    # --- reconciliation --------------------------------------------------
    def _reconcile(self, equity: Optional[float] = None) -> None:
        if self.dry_run:
            return
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        realized = self.execu.reconcile_closed_trades(self.mon, day_start)
        if realized != 0.0 and self.daily is not None:
            self.daily.realized_pnl_today += realized
            if self.daily.loss_fraction() >= self.s.max_daily_loss:
                self.mon.warning(
                    f"Daily loss {self.daily.loss_fraction():.2%} hit limit "
                    f"{self.s.max_daily_loss:.2%} — new entries halted until tomorrow."
                )

    # --- loop ------------------------------------------------------------
    def loop(self) -> None:
        period = _TF_SECONDS.get(self.s.timeframe, 900)
        self.mon.info(f"Entering main loop (scanning {len(self.s.symbols)} symbols "
                      f"every ~{period // 60} min on closed bars).")
        while not self._stop:
            try:
                acct = self.feed.account_info()
                if self.mon.maybe_rollover(equity=acct.equity):
                    self.daily = DailyState(start_equity=acct.equity)
                self._reconcile(equity=acct.equity)
                self.run_pipeline()
                self._write_status(acct)
            except Exception as e:
                self.mon.error(f"Pipeline error: {e}")
            for _ in range(period):
                if self._stop:
                    break
                time.sleep(1)

    def request_stop(self, *_args) -> None:
        self.mon.info("Stop requested — finishing current step then shutting down.")
        self._stop = True


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M5 EMA/RSI/ATR multi-symbol forex bot (DEMO-first).")
    p.add_argument("--once", action="store_true",
                   help="single dry pipeline pass, log only (no orders).")
    p.add_argument("--symbol", default=None,
                   help="watch only this one symbol this run (overrides the watchlist).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    settings = Settings.load()
    if args.symbol:
        import dataclasses
        sym = args.symbol.upper()
        settings = dataclasses.replace(settings, symbol=sym, symbols=[sym])

    bot = Bot(settings, dry_run=args.once)
    signal.signal(signal.SIGINT, bot.request_stop)
    try:
        signal.signal(signal.SIGTERM, bot.request_stop)
    except (ValueError, AttributeError):
        pass

    try:
        bot.start()
        if args.once:
            bot.run_pipeline()
        else:
            bot.loop()
    except Exception as e:
        bot.mon.error(f"Fatal: {e}")
        bot.shutdown()
        return 1
    bot.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
