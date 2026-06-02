"""Module 6 — Execution / Order Management.

Places, modifies, and closes orders on MT5, tracks this bot's open positions, and
reconciles closed deals into the trade ledger.

Safety rails (defense in depth — the risk manager already vetoed unsafe trades,
but execution refuses again here):
  * Refuses to trade if the kill switch is on.
  * Refuses LIVE unless settings.is_live is True.
  * Cross-checks the connected account is not REAL while in DEMO mode.
  * Every market order carries an attached SL and TP.
  * Lot size is normalized to the broker's volume step / min / max.

Only positions tagged with our ``magic_number`` are considered "ours".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from .config import Settings
from .monitor import ClosedTrade
from .risk import TradePlan
from .strategy import PositionSide

try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False


# MT5 deal-entry / deal-type constants (numeric so the pure builder needs no mt5).
_DEAL_ENTRY_IN = 0
_DEAL_ENTRY_OUT = 1
_DEAL_ENTRY_INOUT = 2
_DEAL_ENTRY_OUT_BY = 3
_DEAL_TYPE_BUY = 0


def _iso_from_epoch(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def build_closed_trades_from_deals(
    deals: Iterable,
    magic: int,
    recorded_ids,
    open_map: Optional[dict] = None,
) -> List[ClosedTrade]:
    """Turn a flat list of MT5 deals into completed ClosedTrade rows.

    Pure and MT5-independent (reads attributes via getattr), so it can be unit
    tested with plain stand-in objects. Deals are grouped by ``position_id``; a
    position is "closed" once it has an OUT deal. Net P&L sums profit + commission
    + swap across all of the position's deals (broker truth). ``open_map`` (from
    monitor) supplies stop/target/risk so the R-multiple can be computed.
    """
    open_map = open_map or {}
    recorded = set(recorded_ids or set())

    # Group ALL deals by position first — do NOT magic-filter per deal. A broker
    # TP/SL close writes the OUT deal with magic 0, so filtering every deal would
    # drop it and the trade would never be recorded.
    groups: dict = {}
    for d in deals:
        pid = getattr(d, "position_id", 0) or 0
        groups.setdefault(pid, []).append(d)

    trades: List[ClosedTrade] = []
    for pid, ds in groups.items():
        if not pid or pid in recorded:
            continue
        ins = [d for d in ds if getattr(d, "entry", None) in (_DEAL_ENTRY_IN, _DEAL_ENTRY_INOUT)]
        outs = [d for d in ds if getattr(d, "entry", None) in (_DEAL_ENTRY_OUT, _DEAL_ENTRY_OUT_BY)]
        if not ins or not outs:
            continue  # still open or only partially visible in this window

        # Partial-close safety: a position is only "closed" once the OUT volume has
        # caught up with the IN volume. Without this, a partial take-profit (which
        # writes an OUT deal while the remainder stays open) would be wrongly recorded
        # as a full closed trade. When finally flat, the single recorded row sums ALL
        # deals (partial + final) for the correct net P&L.
        in_vol = sum(float(getattr(d, "volume", 0) or 0) for d in ins)
        out_vol = sum(float(getattr(d, "volume", 0) or 0) for d in outs)
        if out_vol + 1e-9 < in_vol:
            continue  # only partially closed so far — wait until the position is flat

        # Ownership is decided by the OPENING deal's magic (the bot set it on the
        # entry order). magic=None records every closed position regardless.
        if magic is not None and getattr(ins[0], "magic", None) != magic:
            continue

        in_d = ins[0]
        out_d = outs[-1]
        side = "LONG" if getattr(in_d, "type", 0) == _DEAL_TYPE_BUY else "SHORT"
        pnl = sum(
            float(getattr(d, "profit", 0) or 0)
            + float(getattr(d, "commission", 0) or 0)
            + float(getattr(d, "swap", 0) or 0)
            for d in ds
        )
        commission = sum(float(getattr(d, "commission", 0) or 0) for d in ds)
        swap = sum(float(getattr(d, "swap", 0) or 0) for d in ds)

        info = open_map.get(str(pid)) or open_map.get(pid) or {}
        risk = float(info.get("risk_amount", 0) or 0)
        r_multiple = round(pnl / risk, 3) if risk > 0 else 0.0

        trades.append(ClosedTrade(
            open_time_utc=_iso_from_epoch(getattr(in_d, "time", 0)),
            close_time_utc=_iso_from_epoch(getattr(out_d, "time", 0)),
            symbol=getattr(in_d, "symbol", info.get("symbol", "")) or "",
            side=side,
            lots=float(getattr(in_d, "volume", 0) or 0),
            entry=float(getattr(in_d, "price", 0) or 0),
            exit=float(getattr(out_d, "price", 0) or 0),
            stop_loss=float(info.get("stop_loss", 0) or 0),
            take_profit=float(info.get("take_profit", 0) or 0),
            pnl=round(pnl, 2),
            commission=round(commission, 2),
            swap=round(swap, 2),
            r_multiple=r_multiple,
            position_id=int(pid),
            reason_open=info.get("reason_open", ""),
            reason_close="position closed (TP/SL/opposite-cross/manual)",
        ))
    return trades


@dataclass(frozen=True)
class OpenPosition:
    ticket: int
    symbol: str
    side: PositionSide
    lots: float
    entry: float
    stop_loss: float
    take_profit: float
    profit: float
    open_time_utc: str


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    detail: str
    ticket: Optional[int] = None


def _require_mt5() -> None:
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 library not available on this platform.")


def _normalize_volume(volume: float, vmin: float, vmax: float, step: float) -> float:
    if step > 0:
        volume = math.floor(volume / step + 1e-9) * step
    volume = max(vmin, min(volume, vmax))
    return round(volume, 2)


class Executor:
    def __init__(self, settings: Settings, monitor=None) -> None:
        self.s = settings
        self.mon = monitor

    # --- Guards ----------------------------------------------------------
    def _guard(self) -> Optional[str]:
        """Return a refusal reason, or None if trading is permitted."""
        if self.s.kill_switch:
            return "kill switch ON"
        if self.s.live_misconfigured:
            return "LIVE requested but not confirmed"
        if self.s.mode.value == "LIVE" and not self.s.is_live:
            return "LIVE mode without confirmation"
        return None

    # --- Position tracking ----------------------------------------------
    def open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        _require_mt5()
        symbol = symbol or self.s.symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        out: List[OpenPosition] = []
        for p in positions:
            if p.magic != self.s.magic_number:
                continue  # not ours
            side = PositionSide.LONG if p.type == mt5.POSITION_TYPE_BUY else PositionSide.SHORT
            out.append(OpenPosition(
                ticket=p.ticket, symbol=p.symbol, side=side, lots=p.volume,
                entry=p.price_open, stop_loss=p.sl, take_profit=p.tp,
                profit=p.profit,
                open_time_utc=str(p.time),
            ))
        return out

    def count_open(self, symbol: Optional[str] = None) -> int:
        return len(self.open_positions(symbol))

    def open_positions_all(self) -> List[OpenPosition]:
        """All of THIS bot's open positions across every symbol (magic-filtered)."""
        _require_mt5()
        positions = mt5.positions_get()  # no symbol -> all
        if positions is None:
            return []
        out: List[OpenPosition] = []
        for p in positions:
            if p.magic != self.s.magic_number:
                continue
            side = PositionSide.LONG if p.type == mt5.POSITION_TYPE_BUY else PositionSide.SHORT
            out.append(OpenPosition(
                ticket=p.ticket, symbol=p.symbol, side=side, lots=p.volume,
                entry=p.price_open, stop_loss=p.sl, take_profit=p.tp,
                profit=p.profit, open_time_utc=str(p.time),
            ))
        return out

    # --- Orders ----------------------------------------------------------
    def open_position(self, plan: TradePlan, symbol: Optional[str] = None) -> OrderResult:
        refusal = self._guard()
        if refusal:
            res = OrderResult(False, f"refused to open: {refusal}")
            self._log_order(res)
            return res

        _require_mt5()
        symbol = symbol or self.s.symbol
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            res = OrderResult(False, f"no symbol/tick info for {symbol}")
            self._log_order(res)
            return res

        # Spread guard: refuse to enter when the live spread is abnormally wide
        # (news, illiquid hour) — that's where slippage quietly destroys an M5 edge.
        max_sp = getattr(self.s, "max_spread_points", 0)
        point = getattr(info, "point", 0) or 0
        if max_sp and max_sp > 0 and point > 0:
            spread_points = (tick.ask - tick.bid) / point
            if spread_points > max_sp:
                res = OrderResult(
                    False,
                    f"refused to open {symbol}: spread {spread_points:.0f}pts "
                    f"> max {max_sp}pts")
                self._log_order(res)
                return res

        volume = _normalize_volume(plan.lots, info.volume_min, info.volume_max, info.volume_step)
        if plan.side is PositionSide.LONG:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # When trend-riding is on we do NOT hand the broker a take-profit, so the
        # broker won't auto-close at the target; the bot rides the move and exits
        # on the first bar that closes against the trade. The stop-loss is always
        # kept as the hard downside protection.
        broker_tp = 0.0 if getattr(self.s, "ride_trend_after_tp", False) else plan.take_profit

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": plan.stop_loss,
            "tp": broker_tp,
            "deviation": self.s.deviation_points,
            "magic": self.s.magic_number,
            "comment": "forex-bot v1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling(info),
        }
        result = mt5.order_send(request)
        if result is None:
            code, msg = mt5.last_error()
            res = OrderResult(False, f"order_send returned None: ({code}) {msg}")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            res = OrderResult(False, f"retcode={result.retcode} {result.comment}")
        else:
            # Log realized entry slippage (intended plan price vs actual fill) so cost
            # drift between backtest assumptions and live fills is measurable.
            sign = 1.0 if plan.side is PositionSide.LONG else -1.0
            slip = (result.price - plan.entry_price) * sign
            slip_pts = (slip / point) if point > 0 else 0.0
            res = OrderResult(True,
                              f"opened {plan.side.value} {volume} {symbol} @ {result.price} "
                              f"SL {plan.stop_loss} TP {plan.take_profit} "
                              f"(slippage {slip_pts:+.1f}pts)",
                              ticket=result.order)
        self._log_order(res)
        return res

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        """Close a position at market. ``volume=None`` closes the whole position;
        a smaller value does a PARTIAL close (banks part, leaves the rest open with
        the same ticket). Volume is clamped to the position size and normalized."""
        refusal = self._guard()
        if refusal:
            # Kill switch blocks NEW orders, not closes — allow risk-reducing exits.
            if self.s.kill_switch:
                refusal = None
        if refusal:
            res = OrderResult(False, f"refused to close: {refusal}")
            self._log_order(res)
            return res

        _require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            res = OrderResult(False, f"position {ticket} not found")
            self._log_order(res)
            return res
        p = positions[0]
        tick = mt5.symbol_info_tick(p.symbol)
        info = mt5.symbol_info(p.symbol)
        if tick is None or info is None:
            res = OrderResult(False, f"no tick/info to close {ticket}")
            self._log_order(res)
            return res

        if p.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        # Whole close by default; otherwise a normalized partial (never > position).
        if volume is None:
            close_vol = p.volume
        else:
            close_vol = _normalize_volume(min(volume, p.volume), info.volume_min,
                                          info.volume_max, info.volume_step)
        if close_vol <= 0:
            res = OrderResult(False, f"close volume rounded to zero for {ticket}")
            self._log_order(res)
            return res
        partial = close_vol < p.volume

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": close_vol,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.s.deviation_points,
            "magic": self.s.magic_number,
            "comment": "forex-bot v1 partial" if partial else "forex-bot v1 close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling(info),
        }
        result = mt5.order_send(request)
        if result is None:
            code, msg = mt5.last_error()
            res = OrderResult(False, f"close order_send None: ({code}) {msg}")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            res = OrderResult(False, f"close retcode={result.retcode} {result.comment}")
        else:
            tag = f"partial-closed {close_vol}" if partial else "closed"
            res = OrderResult(True, f"{tag} position {ticket} @ {result.price}", ticket=ticket)
        self._log_order(res)
        return res

    def modify_sl_tp(self, ticket: int, *, stop_loss: float, take_profit: float) -> OrderResult:
        _require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(False, f"position {ticket} not found")
        p = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": ticket,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": self.s.magic_number,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            detail = "modify failed" if result is None else f"retcode={result.retcode}"
            res = OrderResult(False, detail, ticket=ticket)
        else:
            res = OrderResult(True, f"modified {ticket} SL {stop_loss} TP {take_profit}", ticket=ticket)
        self._log_order(res)
        return res

    # --- helpers ---------------------------------------------------------
    def _pick_filling(self, info):
        """Choose a filling mode the symbol supports (broker-dependent)."""
        try:
            mode = info.filling_mode
            if mode & 1:  # SYMBOL_FILLING_FOK
                return mt5.ORDER_FILLING_FOK
            if mode & 2:  # SYMBOL_FILLING_IOC
                return mt5.ORDER_FILLING_IOC
        except Exception:
            pass
        return mt5.ORDER_FILLING_IOC

    def _log_order(self, res: OrderResult) -> None:
        if self.mon is not None:
            self.mon.log_order(res.ok, res.detail)

    # --- Trade reconciliation -------------------------------------------
    def find_position_id(self, order_ticket: int, symbol: Optional[str] = None) -> Optional[int]:
        """Resolve the position id for a just-opened order ticket.

        For market deals the position identifier usually equals the order ticket,
        but we confirm via positions_get so the open-trade bookkeeping is keyed
        correctly for later R-multiple calculation.
        """
        _require_mt5()
        pos = mt5.positions_get(ticket=order_ticket)
        if pos:
            return getattr(pos[0], "identifier", pos[0].ticket)
        for p in self.open_positions(symbol):
            return p.ticket
        return order_ticket

    def reconcile_closed_trades(self, monitor, since: datetime) -> float:
        """Find positions that closed since ``since``, write them to the ledger.

        Captures SL/TP exits the broker filled on its own as well as bot-initiated
        closes. Idempotent (deduped by position_id in the monitor). Returns the
        total NET realized P&L of the newly-recorded trades — feed this into the
        daily-loss state so the halt reflects reality.
        """
        _require_mt5()
        to = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(since, to)
        if deals is None or len(deals) == 0:
            return 0.0
        trades = build_closed_trades_from_deals(
            deals,
            self.s.magic_number,
            monitor.recorded_position_ids,
            monitor._read_open_map(),
        )
        realized = 0.0
        for t in trades:
            if monitor.record_trade(t):
                realized += t.pnl
                monitor.pop_open(t.position_id)
        return realized

    def record_closed_position(self, monitor, ticket: int) -> int:
        """Record a single, already-closed position by its ticket — querying that
        position's deals directly and ignoring magic (broker TP/SL closes stamp the
        OUT deal with magic 0). Used to clean up positions the bot still tracks as
        open but that have actually closed. Idempotent. Returns count recorded."""
        _require_mt5()
        try:
            deals = mt5.history_deals_get(position=ticket)
        except Exception:
            deals = None
        if not deals:
            return 0
        trades = build_closed_trades_from_deals(
            deals, None, monitor.recorded_position_ids, monitor._read_open_map())
        n = 0
        for t in trades:
            if monitor.record_trade(t):
                monitor.pop_open(t.position_id)
                n += 1
        return n
