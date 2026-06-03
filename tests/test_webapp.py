"""Tests for the Flask web app API (offline, CSV-backed)."""

import json

import pytest

from webapp.app import create_app
import webapp.app as appmod
from bot.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the app's settings at a temp log dir with a seeded ledger.
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "trades.csv").write_text(
        "close_time_utc,open_time_utc,symbol,side,lots,entry,exit,stop_loss,"
        "take_profit,pnl,commission,swap,r_multiple,position_id,reason_open,reason_close\n"
        "2026-05-30T09:00:00+00:00,2026-05-30T08:00:00+00:00,EURUSD,LONG,0.5,"
        "1.1000,1.1030,1.0985,1.10225,143.0,-6.0,-1.0,1.43,111,o,c\n"
        "2026-05-29T09:00:00+00:00,2026-05-29T08:00:00+00:00,EURUSD,SHORT,0.2,"
        "1.2000,1.2030,1.2030,1.1955,-60.0,-2.0,0.0,-0.6,222,o,c\n",
        encoding="utf-8",
    )
    (logs / "bot.log").write_text("line one\nline two\n", encoding="utf-8")
    (logs / "status.json").write_text(json.dumps({
        "updated_utc": "2026-05-30T09:05:00+00:00", "running": True, "mode": "DEMO",
        "symbol": "EURUSD", "equity": 10000.0, "open_positions": [],
        "daily_realized_pnl": 83.0, "daily_loss_fraction": 0.0,
    }), encoding="utf-8")

    base = Settings.load()
    import dataclasses
    test_settings = dataclasses.replace(base, log_dir=logs)
    monkeypatch.setattr(appmod, "_settings", lambda: test_settings)
    # never touch a live MT5 terminal in tests
    monkeypatch.setattr(appmod, "_live_mt5", lambda: None)
    monkeypatch.setattr(appmod, "_live_scan", lambda: None)
    monkeypatch.setattr(appmod, "_reconcile_closed", lambda: 0)

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Forex Bot" in r.data


def test_status_endpoint(client):
    r = client.get("/api/status")
    d = r.get_json()
    assert d["config"]["symbol"] == "EURUSD"
    assert d["snapshot"]["equity"] == 10000.0
    assert d["bot_running"] is False
    assert d["kill_switch"] is False


def test_trades_all(client):
    d = client.get("/api/trades?period=all").get_json()
    assert d["summary"]["n_trades"] == 2
    assert d["summary"]["net_pnl"] == pytest.approx(83.0)
    assert len(d["trades"]) == 2
    # newest trade first (latest close on top)
    assert d["trades"][0]["close_time_utc"] > d["trades"][1]["close_time_utc"]


def test_trades_today_filter(client):
    d = client.get("/api/trades?period=today").get_json()
    assert "summary" in d  # count depends on clock; just ensure it responds


def test_trades_bad_period(client):
    r = client.get("/api/trades?period=fortnight")
    assert r.status_code == 400


def test_kill_toggle(client):
    on = client.post("/api/control/kill", json={"on": True}).get_json()
    assert on["kill_switch"] is True
    off = client.post("/api/control/kill", json={"on": False}).get_json()
    assert off["kill_switch"] is False


def test_log_endpoint(client):
    d = client.get("/api/log?lines=10").get_json()
    assert d["lines"][-1] == "line two"


def test_bot_control_bad_action(client):
    r = client.post("/api/control/bot", json={"action": "frobnicate"})
    assert r.status_code == 400


def test_status_open_positions_fallback_to_file(client):
    # When MT5 is offline and status has no positions, open_trades.json is shown.
    logs = appmod._settings().log_dir
    (logs / "open_trades.json").write_text(json.dumps({
        "999": {"symbol": "EURUSD", "side": "LONG", "stop_loss": 1.16,
                "take_profit": 1.17, "risk_amount": 10.0, "reason_open": "x"}
    }), encoding="utf-8")
    d = client.get("/api/status").get_json()
    pos = d["snapshot"]["open_positions"]
    assert len(pos) == 1 and pos[0]["symbol"] == "EURUSD" and pos[0]["bot"] is True


def test_scan_empty_when_offline(client):
    d = client.get("/api/scan").get_json()
    assert d["rows"] == []


def test_reconcile_closed_noop_offline():
    # With MetaTrader5 absent, reconciliation must be a harmless no-op.
    appmod._recon_state["ts"] = 0.0
    assert appmod._reconcile_closed() == 0
