"""Flask web app — live dashboard + controls for the forex bot.

Run:
  uv run python -m webapp.app                # http://127.0.0.1:5000
  uv run python -m webapp.app --port 8000

Data sources (per the chosen design):
  * logs/status.json   — live snapshot the bot writes each loop (equity, open
                         positions, last signal, daily P&L, kill state).
  * logs/trades.csv     — closed-trade ledger (history, filters, charts).
  * logs/bot.log        — recent activity for the log viewer.
  * logs/control.json   — runtime kill switch (web app writes, bot polls).
  * MetaTrader5 (best-effort) — if importable AND the terminal is up, the status
                         endpoint augments the snapshot with a real-time account
                         read so equity is current between bot loops.

Controls:
  * POST /api/control/kill   {"on": true|false}   → toggle the live kill switch.
  * POST /api/control/bot    {"action":"start"|"stop"} → launch/terminate the
                         bot as a subprocess (handle held in this process).

SAFETY: defaults bind to 127.0.0.1 (localhost only). Do not expose this to a
network without adding authentication — the control endpoints can affect trading.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from bot.config import Settings
from bot.control import read_control, write_control
from bot.report import build_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MetaTrader5's Python client is process-global; serialize the web app's
# best-effort initialize()/shutdown() cycles so concurrent requests (status +
# scan) can't race each other.
_MT5_LOCK = threading.Lock()

# Cache the (relatively expensive) live watchlist scan so the dashboard's 5s poll
# doesn't re-scan the whole watchlist every tick.
_LIVE_SCAN_TTL_SEC = 60.0
_live_scan_cache: dict = {"ts": 0.0, "payload": None}

# How often the web app pulls closed deals from MT5 to backfill trades.csv, and
# how far back it looks (a wide window is safe — recording is deduped by id).
_RECON_TTL_SEC = 30.0
_RECON_LOOKBACK_DAYS = 365
_recon_state: dict = {"ts": 0.0}

# Per-symbol cache for charting any watchlist symbol on demand (the bot only
# writes candles.json for its single active symbol).
_LIVE_CANDLES_TTL_SEC = 10.0
_live_candles_cache: dict = {}  # symbol -> {"ts": float, "payload": dict}


# --- Forex session helpers (used for the market open/closed banner) ----------
import datetime as _dt

_FOREX_OPEN_HOUR_UTC = 22  # standard: opens ~Sun 22:00 UTC, closes ~Fri 22:00 UTC


def _next_forex_open(now: _dt.datetime) -> _dt.datetime:
    """Next Sunday 22:00 UTC at or after ``now`` (the weekly reopen)."""
    days_until_sun = (6 - now.weekday()) % 7  # Monday=0 ... Sunday=6
    cand = (now + _dt.timedelta(days=days_until_sun)).replace(
        hour=_FOREX_OPEN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if cand <= now:
        cand += _dt.timedelta(days=7)
    return cand


def _market_status(log_dir, live, snap) -> dict:
    """Decide if the market is open. Live bid/ask spread is the strongest signal
    (frozen bid==ask => closed); fall back to candle freshness when offline."""
    now = datetime.now(timezone.utc)
    cj = _read_json(Path(log_dir) / "candles.json")
    cs = cj.get("candles") or []
    last_t = cs[-1]["t"] if cs else None

    if live and live.get("bid") is not None and live.get("ask") is not None:
        is_open = float(live["bid"]) != float(live["ask"])
    elif last_t is not None:
        is_open = (now.timestamp() - float(last_t)) < 1200  # candle < 20 min old
    else:
        is_open = False

    nxt = _next_forex_open(now)
    return {
        "open": bool(is_open),
        "now_utc": now.isoformat(timespec="seconds"),
        "last_candle_utc": (datetime.fromtimestamp(last_t, tz=timezone.utc).isoformat()
                            if last_t else None),
        "next_open_utc": nxt.isoformat(),
        "seconds_to_open": max(0, int((nxt - now).total_seconds())),
        "note": "Forex trades ~24/5: opens Sunday and closes Friday evening (UTC). Times approximate / broker-dependent.",
    }


def _settings() -> Settings:
    return Settings.load()


def _log_dir() -> Path:
    return _settings().log_dir


# Held in this process so start/stop is reliable for a single local user.
_bot_proc: "subprocess.Popen | None" = None


def _bot_running() -> bool:
    global _bot_proc
    return _bot_proc is not None and _bot_proc.poll() is None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_live_positions(mt5, s: Settings) -> list[dict]:
    """All open positions straight from the terminal, enriched with the bot's own
    bookkeeping (planned SL/TP/risk) from open_trades.json when present.

    Intentionally NOT magic-filtered: we want every position the account is
    holding to be visible on the dashboard, and we tag which ones are this bot's.
    """
    try:
        raw = mt5.positions_get()
    except Exception:
        raw = None
    if not raw:
        return []
    ctx = _read_json(s.log_dir / "open_trades.json")
    out: list[dict] = []
    for p in raw:
        info = (ctx.get(str(p.ticket))
                or ctx.get(str(getattr(p, "identifier", ""))) or {})
        side = "LONG" if p.type == getattr(mt5, "POSITION_TYPE_BUY", 0) else "SHORT"
        out.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "side": side,
            "lots": p.volume,
            "entry": p.price_open,
            "stop_loss": p.sl or info.get("stop_loss"),
            "take_profit": p.tp or info.get("take_profit"),
            "profit": p.profit,
            "bot": bool(p.magic == s.magic_number),
        })
    return out


def _positions_from_file(s: Settings) -> list[dict]:
    """Fallback when MT5 is unreachable: show the bot's tracked open trades from
    open_trades.json (no live P&L, but at least the positions are visible)."""
    data = _read_json(s.log_dir / "open_trades.json")
    out: list[dict] = []
    for pid, info in (data or {}).items():
        out.append({
            "ticket": pid,
            "symbol": info.get("symbol", ""),
            "side": info.get("side", ""),
            "lots": None,
            "entry": None,
            "stop_loss": info.get("stop_loss"),
            "take_profit": info.get("take_profit"),
            "profit": None,
            "bot": True,
        })
    return out


def _live_mt5() -> dict | None:
    """Best-effort real-time read straight from MT5 (if available): account,
    current tick, and ALL open positions. Returns None when MT5 can't be reached."""
    try:
        from bot.data import DataFeed, _MT5_AVAILABLE
        import MetaTrader5 as mt5  # type: ignore
        if not _MT5_AVAILABLE:
            return None
        s = _settings()
        with _MT5_LOCK:
            feed = DataFeed(s)
            acct = feed.connect()
            try:
                tick = None
                try:
                    tick = feed.get_tick()
                except Exception:
                    pass
                positions = _read_live_positions(mt5, s)
            finally:
                feed.shutdown()
        return {
            "equity": acct.equity, "balance": acct.balance,
            "currency": acct.currency, "trade_mode": acct.trade_mode,
            "bid": getattr(tick, "bid", None), "ask": getattr(tick, "ask", None),
            "open_positions": positions,
        }
    except Exception:
        return None


def _live_scan() -> dict | None:
    """Best-effort live watchlist scan computed by the web app itself (cached for
    ``_LIVE_SCAN_TTL_SEC``). Mirrors the bot's _write_scan payload and persists to
    scan.json so the result is shared. Returns None when MT5 is unavailable."""
    now = time.time()
    cached = _live_scan_cache.get("payload")
    if cached is not None and (now - _live_scan_cache.get("ts", 0.0)) < _LIVE_SCAN_TTL_SEC:
        return cached
    try:
        from bot.data import DataFeed, _MT5_AVAILABLE
        from bot.scanner import scan_watchlist, rank_opportunities
        from bot.strategy import PositionSide
        import MetaTrader5 as mt5  # type: ignore
        if not _MT5_AVAILABLE:
            return None
        s = _settings()
        with _MT5_LOCK:
            feed = DataFeed(s)
            feed.connect()
            try:
                open_map: dict = {}
                try:
                    for p in (mt5.positions_get() or []):
                        open_map[p.symbol] = (
                            PositionSide.LONG
                            if p.type == getattr(mt5, "POSITION_TYPE_BUY", 0)
                            else PositionSide.SHORT)
                except Exception:
                    pass
                scans = scan_watchlist(feed, s, open_symbols=open_map)
                ranked = rank_opportunities(scans)
            finally:
                feed.shutdown()
        rows = [sc.to_dict() for sc in sorted(
            scans, key=lambda x: (x.tradable, x.score), reverse=True)]
        active = (next(iter(open_map), None)
                  or (ranked[0].symbol if ranked else s.symbol))
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timeframe": s.timeframe, "htf_timeframe": s.htf_timeframe,
            "scan_mode": s.scan_mode, "active_symbol": active,
            "top": ranked[0].symbol if ranked else None,
            "rows": rows, "source": "webapp-live",
        }
        try:
            (s.log_dir / "scan.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        _live_scan_cache["payload"] = payload
        _live_scan_cache["ts"] = now
        return payload
    except Exception:
        return None


def _live_candles(symbol: str, n: int = 180) -> dict | None:
    """Candles + EMAs for ANY symbol, computed live from MT5 (cached per symbol).
    Mirrors the bot's _write_candles payload so the chart renders identically.
    Returns None when MT5 is unavailable or the broker doesn't offer the symbol."""
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return None
    now = time.time()
    c = _live_candles_cache.get(symbol)
    if c and (now - c["ts"]) < _LIVE_CANDLES_TTL_SEC:
        return c["payload"]
    try:
        import pandas as pd
        from bot.data import DataFeed, _MT5_AVAILABLE
        from bot.indicators import add_indicators
        if not _MT5_AVAILABLE:
            return None
        s = _settings()
        candles = None
        with _MT5_LOCK:
            feed = DataFeed(s)
            feed.connect()
            try:
                if hasattr(feed, "ensure_symbol") and not feed.ensure_symbol(symbol):
                    return None
                candles = feed.get_candles(symbol=symbol, timeframe=s.timeframe,
                                           n=s.history_bars)
            finally:
                feed.shutdown()
        enriched = add_indicators(candles, ema_fast=s.ema_fast, ema_slow=s.ema_slow,
                                  rsi_period=s.rsi_period, atr_period=s.atr_period)
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
        payload = {"symbol": symbol, "timeframe": s.timeframe,
                   "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "candles": out, "ema_fast": ef, "ema_slow": es, "source": "webapp-live"}
        _live_candles_cache[symbol] = {"ts": now, "payload": payload}
        return payload
    except Exception:
        return None


def _record_position_close(ticket: int, attempts: int = 5, delay: float = 0.5) -> int:
    """Record a just-closed position into trades.csv immediately, by querying that
    exact position's deals (retrying briefly while the broker propagates the close)
    and recording regardless of magic. Returns the number of trades recorded."""
    try:
        from bot.data import DataFeed, _MT5_AVAILABLE
        from bot.execution import build_closed_trades_from_deals
        from bot.monitor import Monitor
        import MetaTrader5 as mt5  # type: ignore
        if not _MT5_AVAILABLE:
            return 0
        s = _settings()
        for _ in range(max(1, attempts)):
            with _MT5_LOCK:
                feed = DataFeed(s)
                feed.connect()
                try:
                    deals = mt5.history_deals_get(position=ticket)
                finally:
                    feed.shutdown()
            if deals:
                mon = Monitor(s.log_dir, console=False)
                trades = build_closed_trades_from_deals(
                    deals, None, mon.recorded_position_ids, mon._read_open_map())
                recorded = 0
                for tr in trades:
                    if mon.record_trade(tr):
                        mon.pop_open(tr.position_id)
                        recorded += 1
                if recorded:
                    return recorded
            time.sleep(delay)
    except Exception:
        return 0
    return 0


def _reconcile_closed() -> int:
    """Pull closed deals from MT5 and append any not-yet-recorded ones to
    trades.csv, so the history + P&L fills in even when the bot isn't running.

    Reuses the bot's exact reconciliation logic (magic-filtered, deduped by
    position id), so the web app and the bot can both run it without
    double-recording. Throttled to once per ``_RECON_TTL_SEC``. No-op when MT5 is
    unavailable. Returns the number of newly recorded trades (0 on no-op/error).
    """
    now = time.time()
    if (now - _recon_state.get("ts", 0.0)) < _RECON_TTL_SEC:
        return 0
    _recon_state["ts"] = now  # claim the slot up-front so polls don't pile up
    try:
        from bot.data import DataFeed, _MT5_AVAILABLE
        from bot.execution import Executor
        from bot.monitor import Monitor
        if not _MT5_AVAILABLE:
            return 0
        s = _settings()
        since = datetime.now(timezone.utc) - timedelta(days=_RECON_LOOKBACK_DAYS)
        with _MT5_LOCK:
            feed = DataFeed(s)
            feed.connect()
            try:
                # Fresh Monitor each call: it re-reads trades.csv so its dedup set
                # already includes anything the bot recorded.
                mon = Monitor(s.log_dir, console=False)
                execu = Executor(s, monitor=mon)
                before = mon.stats.n_trades
                execu.reconcile_closed_trades(mon, since)
                return max(0, mon.stats.n_trades - before)
            finally:
                feed.shutdown()
    except Exception:
        return 0


def _json_safe(obj):
    """Recursively replace NaN / Infinity with None so the payload is STRICT JSON.
    Flask's jsonify emits bare ``NaN``/``Infinity`` tokens, which are invalid JSON
    and make the browser's JSON.parse throw -- blanking the whole trades table when
    even one field (e.g. an empty reason_open on a backfilled trade) is NaN.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # numpy scalars / pandas NaT and the like: catch any self-unequal NaN value
    try:
        if obj != obj:  # noqa: PLR0124  (NaN is the only value != itself)
            return None
    except Exception:
        pass
    return obj


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/status")
    def api_status():
        s = _settings()
        snap = _read_json(s.log_dir / "status.json")
        live = _live_mt5()  # may be None (offline / non-Windows / terminal down)
        ctrl = read_control(s.log_dir)
        market = _market_status(s.log_dir, live, snap)

        # Open positions: prefer a live MT5 read, then the bot's last snapshot,
        # then the tracked-trades file — so they show regardless of loop state.
        if live and live.get("open_positions") is not None:
            open_positions = live["open_positions"]
        elif snap.get("open_positions"):
            open_positions = snap["open_positions"]
        else:
            open_positions = _positions_from_file(s)
        snap = {**snap, "open_positions": open_positions}

        return jsonify({
            "now_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {
                "symbol": s.symbol, "timeframe": s.timeframe,
                "mode": "LIVE" if s.is_live else "DEMO",
                "risk_per_trade": s.risk_per_trade,
                "max_daily_loss": s.max_daily_loss,
                "symbols": s.symbols,
                "watchlist_size": len(s.symbols),
                "scan_mode": s.scan_mode,
                "htf_timeframe": s.htf_timeframe,
            },
            "snapshot": snap,                 # last bot loop
            "live": live,                     # real-time MT5 (or null)
            "kill_switch": bool(ctrl.get("kill_switch", False)),
            "bot_running": _bot_running(),
            "market": market,
        })

    @app.route("/api/trades")
    def api_trades():
        s = _settings()
        _reconcile_closed()  # backfill newly-closed trades from MT5 (throttled)
        period = request.args.get("period", "all")
        frm = request.args.get("from") or None
        to = request.args.get("to") or None
        try:
            summary, rows = build_report(s.log_dir / "trades.csv", period, frm=frm, to=to)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            # Never let a bad ledger blank the whole panel with a 500 — return an
            # empty (but valid) payload with a visible note so the UI still renders.
            return jsonify({"summary": {}, "trades": [],
                            "error": f"could not read trade history: {e}"})
        trades = []
        if not rows.empty:
            r = rows.copy()
            if "close_time_utc" in r.columns:
                # Newest trade first so the latest close sits on top of the table.
                r = r.sort_values("close_time_utc", ascending=False)
                r["close_time_utc"] = r["close_time_utc"].astype(str)
            if "open_time_utc" in r.columns:
                r["open_time_utc"] = r["open_time_utc"].astype(str)
            trades = r.to_dict(orient="records")
        return jsonify(_json_safe({"summary": summary.__dict__, "trades": trades}))

    @app.route("/api/candles")
    def api_candles():
        s = _settings()
        want = (request.args.get("symbol") or "").upper().strip()
        if want:
            # A specific symbol was requested — fetch it live from MT5.
            live = _live_candles(want)
            if live and live.get("candles"):
                return jsonify(live)
            # Fall back to the persisted file if it happens to be the same symbol.
            data = _read_json(s.log_dir / "candles.json")
            if data and (data.get("symbol", "").upper() == want):
                return jsonify(data)
            return jsonify({"symbol": want, "timeframe": s.timeframe,
                            "candles": [], "ema_fast": [], "ema_slow": [],
                            "note": "no live data (MT5 offline or symbol not offered)"})
        # No symbol param: serve the bot's active-symbol candles (default view).
        data = _read_json(s.log_dir / "candles.json")
        if not data:
            data = {"symbol": s.symbol, "timeframe": s.timeframe,
                    "candles": [], "ema_fast": [], "ema_slow": []}
        return jsonify(data)

    @app.route("/api/scan")
    def api_scan():
        s = _settings()
        persisted = _read_json(s.log_dir / "scan.json")
        live = _live_scan()  # web app's own MT5 scan (cached); None when offline

        def _ts(d):
            return (d or {}).get("updated_utc") or ""

        # Serve whichever scan is freshest. The live read (when available) keeps
        # the panel populated even if the bot loop never wrote scan.json.
        candidates = [c for c in (live, persisted) if c and c.get("rows")]
        data = max(candidates, key=_ts) if candidates else None
        if not data:
            data = {"rows": [], "active_symbol": s.symbol, "top": None,
                    "scan_mode": s.scan_mode, "timeframe": s.timeframe,
                    "htf_timeframe": s.htf_timeframe, "updated_utc": None}
        return jsonify(data)

    @app.route("/api/log")
    def api_log():
        s = _settings()
        n = min(int(request.args.get("lines", 200)), 2000)
        p = s.log_dir / "bot.log"
        if not p.exists():
            return jsonify({"lines": []})
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        return jsonify({"lines": lines[-n:]})

    @app.route("/api/control/kill", methods=["POST"])
    def api_kill():
        on = bool((request.get_json(silent=True) or {}).get("on", False))
        state = write_control(_log_dir(), kill_switch=on)
        return jsonify({"kill_switch": bool(state.get("kill_switch", False))})

    @app.route("/api/control/close", methods=["POST"])
    def api_close():
        """Manually close a position at market by ticket. The kill switch does NOT
        block this (closing reduces risk). After closing we kick a reconcile so the
        trade lands in the history/ledger right away."""
        body = request.get_json(silent=True) or {}
        ticket = body.get("ticket")
        if ticket in (None, ""):
            return jsonify({"ok": False, "error": "ticket required"}), 400
        try:
            ticket = int(ticket)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"bad ticket: {ticket!r}"}), 400
        try:
            from bot.data import DataFeed, _MT5_AVAILABLE
            from bot.execution import Executor
            if not _MT5_AVAILABLE:
                return jsonify({"ok": False, "error": "MT5 not available on this host"}), 503
            s = _settings()
            with _MT5_LOCK:
                feed = DataFeed(s)
                feed.connect()
                try:
                    res = Executor(s).close_position(ticket)
                finally:
                    feed.shutdown()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        # Record the now-closed trade into history immediately (direct, retried,
        # magic-agnostic). The read-time dedup in report.load_trades collapses any
        # duplicate the bot's own reconcile might also write for this position.
        recorded = 0
        if res.ok:
            try:
                recorded = _record_position_close(ticket)
            except Exception:
                pass
        return jsonify({"ok": bool(res.ok), "detail": res.detail,
                        "ticket": ticket, "recorded": recorded})

    @app.route("/api/control/open", methods=["POST"])
    def api_open():
        """Manually open a position at market from the dashboard (mirror of close).

        Body: {"symbol": "EURUSD", "side": "BUY"|"SELL",
               "lots"?: float, "sl"?: price, "tp"?: price}

        Sizing/stops default to the SAME risk engine the bot uses (ATR stop, fixed
        risk %, RR target) so a manual trade carries identical protection — every
        order still gets an attached stop-loss. Optional ``lots`` overrides the
        risk-sized volume; optional ``sl``/``tp`` override the computed levels.
        The trade is tagged with the bot's magic number and written to the open-
        trade ledger, so if the bot is running it manages the exit (trend-ride /
        partial / opposite-cross) exactly like an auto entry. The live kill switch
        DOES block this — opening adds risk (unlike closing, which reduces it)."""
        import dataclasses
        body = request.get_json(silent=True) or {}
        symbol = str(body.get("symbol") or "").strip().upper()
        side_s = str(body.get("side") or "").strip().upper()
        if not symbol:
            return jsonify({"ok": False, "error": "symbol required"}), 400
        if side_s not in ("BUY", "SELL"):
            return jsonify({"ok": False, "error": "side must be BUY or SELL"}), 400

        def _optf(name):
            v = body.get(name)
            if v in (None, ""):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        lots_override, sl_override, tp_override = _optf("lots"), _optf("sl"), _optf("tp")

        try:
            from bot.data import DataFeed, _MT5_AVAILABLE
            from bot.execution import Executor
            from bot.indicators import add_indicators
            from bot.strategy import Action, PositionSide, Signal
            from bot.risk import RiskManager, DailyState, TradePlan
            from bot.monitor import Monitor
            import math
            if not _MT5_AVAILABLE:
                return jsonify({"ok": False, "error": "MT5 not available on this host"}), 503
            s = _settings()
            # Opening adds risk → honour the live kill switch (close does not).
            if bool(read_control(s.log_dir).get("kill_switch", False)):
                return jsonify({"ok": False, "error": "kill switch is ON — new orders blocked"}), 409

            with _MT5_LOCK:
                feed = DataFeed(s)
                feed.connect()
                try:
                    if not feed.ensure_symbol(symbol):
                        return jsonify({"ok": False, "error": f"{symbol} not offered by broker"}), 400
                    candles = feed.get_candles(symbol=symbol, timeframe=s.timeframe,
                                               n=s.history_bars)
                    enriched = add_indicators(candles, ema_fast=s.ema_fast, ema_slow=s.ema_slow,
                                              rsi_period=s.rsi_period, atr_period=s.atr_period,
                                              adx_period=s.adx_period)
                    last = enriched.iloc[-1]
                    atr = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0
                    price = float(last["close"])
                    if atr <= 0 and lots_override is None and sl_override is None:
                        return jsonify({"ok": False, "error": "ATR not ready — pass explicit lots+sl"}), 409
                    spec = feed.symbol_spec(symbol)
                    acct = feed.account_info()
                    side = PositionSide.LONG if side_s == "BUY" else PositionSide.SHORT
                    sign = 1.0 if side is PositionSide.LONG else -1.0

                    if lots_override is not None:
                        # Manual size: build the plan directly (still attach a stop).
                        stop_distance = (abs(price - sl_override) if sl_override is not None
                                         else atr * s.atr_multiplier)
                        if stop_distance <= 0:
                            return jsonify({"ok": False, "error": "stop distance is zero"}), 409
                        stop_loss = (sl_override if sl_override is not None
                                     else price - sign * stop_distance)
                        take_profit = (tp_override if tp_override is not None
                                       else price + sign * stop_distance * s.risk_reward)
                        plan = TradePlan(
                            side=side, entry_price=round(price, spec.digits),
                            stop_loss=round(stop_loss, spec.digits),
                            take_profit=round(take_profit, spec.digits),
                            lots=round(lots_override, 2),
                            risk_amount=round(lots_override * spec.contract_size * stop_distance, 2),
                            stop_distance=round(stop_distance, spec.digits),
                            reason=f"manual {side.value} {round(lots_override, 2)} lots (dashboard)")
                    else:
                        # Risk-engine sizing: same path as an auto entry.
                        sig = Signal(action=Action.BUY if side_s == "BUY" else Action.SELL,
                                     reason="manual order (dashboard)", price=price, atr=atr,
                                     rsi=float(last["rsi"]) if not math.isnan(last["rsi"]) else 50.0,
                                     ema_fast=float(last["ema_fast"]), ema_slow=float(last["ema_slow"]),
                                     bar_time=enriched.index[-1])
                        rm = RiskManager(
                            risk_per_trade=s.risk_per_trade, max_risk_per_trade=s.max_risk_per_trade,
                            max_daily_loss=s.max_daily_loss, risk_mode=s.risk_mode,
                            atr_multiplier=s.atr_multiplier, risk_reward=s.risk_reward,
                            symbol_spec=spec)
                        # open_positions=0: a deliberate manual trade isn't blocked by the
                        # auto position-count cap, but the risk %/min-lot veto still applies.
                        dec = rm.evaluate(sig, equity=acct.equity, open_positions=0,
                                          daily=DailyState(acct.equity), kill_switch=False,
                                          is_live=s.is_live, allow_live=s.is_live)
                        if not (dec.approved and dec.plan is not None):
                            return jsonify({"ok": False, "error": f"risk veto: {dec.reason}"}), 409
                        plan = dec.plan
                        if sl_override is not None or tp_override is not None:
                            plan = dataclasses.replace(
                                plan,
                                stop_loss=round(sl_override, spec.digits) if sl_override is not None else plan.stop_loss,
                                take_profit=round(tp_override, spec.digits) if tp_override is not None else plan.take_profit)

                    mon = Monitor(s.log_dir, console=False)
                    execu = Executor(s, monitor=mon)
                    res = execu.open_position(plan, symbol=symbol)
                    if res.ok and res.ticket is not None:
                        try:
                            pid = execu.find_position_id(res.ticket, symbol=symbol)
                            mon.note_open(pid, {
                                "symbol": symbol, "side": plan.side.value,
                                "stop_loss": plan.stop_loss, "take_profit": plan.take_profit,
                                "stop_distance": plan.stop_distance, "entry": plan.entry_price,
                                "risk_amount": plan.risk_amount, "reason_open": plan.reason,
                                "open_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            })
                        except Exception:
                            pass
                finally:
                    feed.shutdown()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": bool(res.ok), "detail": res.detail, "ticket": res.ticket,
                        "plan": plan.reason})

    @app.route("/api/control/bot", methods=["POST"])
    def api_bot():
        global _bot_proc
        action = (request.get_json(silent=True) or {}).get("action", "")
        if action == "start":
            if _bot_running():
                return jsonify({"bot_running": True, "note": "already running"})
            _bot_proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return jsonify({"bot_running": _bot_running()})
        if action == "stop":
            if _bot_proc is not None and _bot_running():
                _bot_proc.terminate()
                try:
                    _bot_proc.wait(timeout=10)
                except Exception:
                    _bot_proc.kill()
            _bot_proc = None
            return jsonify({"bot_running": False})
        return jsonify({"error": "action must be 'start' or 'stop'"}), 400

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forex bot web dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default localhost)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app = create_app()
    print(f"Forex bot dashboard → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# end of webapp/app.py
