"""Tests for the report module's period filtering and metric summary."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.report import filter_period, summarize, build_report


NOW = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


def _df(rows):
    # rows: list of (close_time_str, pnl, r_multiple)
    return pd.DataFrame({
        "close_time_utc": pd.to_datetime([r[0] for r in rows], utc=True),
        "open_time_utc": pd.to_datetime([r[0] for r in rows], utc=True),
        "symbol": ["EURUSD"] * len(rows),
        "side": ["LONG"] * len(rows),
        "lots": [0.1] * len(rows),
        "entry": [1.1] * len(rows),
        "exit": [1.1] * len(rows),
        "pnl": [r[1] for r in rows],
        "commission": [0.0] * len(rows),
        "swap": [0.0] * len(rows),
        "r_multiple": [r[2] for r in rows],
    })


SAMPLE = _df([
    ("2026-05-31T09:00:00Z", 100.0, 1.0),   # today
    ("2026-05-30T09:00:00Z", -50.0, -0.5),  # yesterday (1 day ago)
    ("2026-05-29T09:00:00Z", 30.0, 0.3),    # 2 days ago
    ("2026-05-25T09:00:00Z", -20.0, -0.2),  # 6 days ago
    ("2026-05-10T09:00:00Z", 200.0, 2.0),   # 21 days ago
    ("2026-03-01T09:00:00Z", 75.0, 0.7),    # months ago
])


def test_today_filter():
    out = filter_period(SAMPLE, "today", now=NOW)
    assert len(out) == 1
    assert out.iloc[0]["pnl"] == 100.0


def test_yesterday_filter():
    out = filter_period(SAMPLE, "yesterday", now=NOW)
    assert len(out) == 1
    assert out.iloc[0]["pnl"] == -50.0


def test_2d_rolling():
    out = filter_period(SAMPLE, "2d", now=NOW)
    # last 48h: today (9:00) and yesterday (9:00) are within 48h; 2-days-ago 9:00 is >48h
    assert set(out["pnl"]) == {100.0, -50.0}


def test_3d_rolling():
    out = filter_period(SAMPLE, "3d", now=NOW)
    assert set(out["pnl"]) == {100.0, -50.0, 30.0}


def test_week_filter():
    out = filter_period(SAMPLE, "week", now=NOW)
    assert set(out["pnl"]) == {100.0, -50.0, 30.0, -20.0}


def test_month_filter():
    out = filter_period(SAMPLE, "month", now=NOW)
    assert set(out["pnl"]) == {100.0, -50.0, 30.0, -20.0, 200.0}


def test_all_filter():
    out = filter_period(SAMPLE, "all", now=NOW)
    assert len(out) == 6


def test_custom_range_inclusive():
    out = filter_period(SAMPLE, "custom", now=NOW, frm="2026-05-25", to="2026-05-30")
    # inclusive 25->30 covers 05-30 (-50), 05-29 (30), 05-25 (-20)
    assert set(out["pnl"]) == {-50.0, 30.0, -20.0}


def test_unknown_period_raises():
    with pytest.raises(ValueError):
        filter_period(SAMPLE, "fortnight", now=NOW)


def test_summarize_metrics():
    s = summarize(SAMPLE, "all")
    assert s.n_trades == 6
    assert s.wins == 4 and s.losses == 2
    assert s.net_pnl == pytest.approx(335.0)       # 100-50+30-20+200+75
    assert s.gross_profit == pytest.approx(405.0)  # 100+30+200+75
    assert s.gross_loss == pytest.approx(70.0)     # 50+20
    assert s.profit_factor == pytest.approx(405.0 / 70.0)
    assert s.best_trade == 200.0 and s.worst_trade == -50.0


def test_summarize_empty():
    s = summarize(pd.DataFrame(), "today")
    assert s.n_trades == 0 and s.net_pnl == 0.0


def test_max_drawdown():
    # Sequence sorted by time: +100, -50, +30, -20, +200, +75 (months order differs,
    # but summarize uses row order). Construct a clear DD case.
    df = _df([
        ("2026-05-01T00:00:00Z", 100.0, 1.0),
        ("2026-05-02T00:00:00Z", -80.0, -0.8),
        ("2026-05-03T00:00:00Z", -40.0, -0.4),
        ("2026-05-04T00:00:00Z", 30.0, 0.3),
    ])
    s = summarize(df, "all")
    # cum: 100, 20, -20, 10 ; peak 100 → trough -20 → DD = 120
    assert s.max_drawdown == pytest.approx(120.0)


def test_build_report_missing_file_is_empty(tmp_path):
    summary, rows = build_report(tmp_path / "nope.csv", "all", now=NOW)
    assert summary.n_trades == 0
    assert rows.empty
