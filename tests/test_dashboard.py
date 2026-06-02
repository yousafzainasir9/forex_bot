"""Tests for the HTML dashboard generator (data extraction + injection)."""

import json
import re
from pathlib import Path

from bot.dashboard import build_html, _rows_for_web


def _write_csv(p: Path):
    p.write_text(
        "close_time_utc,open_time_utc,symbol,side,lots,entry,exit,stop_loss,"
        "take_profit,pnl,commission,swap,r_multiple,position_id,reason_open,reason_close\n"
        "2026-05-30T09:00:00+00:00,2026-05-30T08:00:00+00:00,EURUSD,LONG,0.5,"
        "1.1000,1.1030,1.0985,1.10225,143.0,-6.0,-1.0,1.43,111,o,c\n"
        "2026-05-29T09:00:00+00:00,2026-05-29T08:00:00+00:00,EURUSD,SHORT,0.2,"
        "1.2000,1.2030,1.2030,1.1955,-60.0,-2.0,0.0,-0.6,222,o,c\n",
        encoding="utf-8",
    )


def test_rows_for_web_shapes(tmp_path):
    csv = tmp_path / "trades.csv"
    _write_csv(csv)
    rows = _rows_for_web(csv)
    assert len(rows) == 2
    assert rows[0]["symbol"] == "EURUSD"
    assert rows[0]["pnl"] == 143.0
    assert "close_time" in rows[0] and rows[0]["close_time"].startswith("2026-05-30")


def test_build_html_injects_valid_json(tmp_path):
    csv = tmp_path / "trades.csv"
    _write_csv(csv)
    html = build_html(csv)
    assert "/*__DATA__*/" not in html          # placeholder was replaced
    assert "chart.umd.min.js" in html          # chart lib referenced
    m = re.search(r"const PAYLOAD = (\{.*?\});", html, re.S)
    assert m
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(data["trades"]) == 2


def test_build_html_empty_ledger(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text(
        "close_time_utc,open_time_utc,symbol,side,lots,entry,exit,stop_loss,"
        "take_profit,pnl,commission,swap,r_multiple,position_id,reason_open,reason_close\n",
        encoding="utf-8",
    )
    html = build_html(csv)
    m = re.search(r"const PAYLOAD = (\{.*?\});", html, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["trades"] == []
