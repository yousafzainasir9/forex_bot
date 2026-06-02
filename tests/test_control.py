"""Tests for the runtime control flag module."""

from bot.control import read_control, write_control, kill_active


def test_default_when_missing(tmp_path):
    assert read_control(tmp_path) == {"kill_switch": False}
    assert kill_active(tmp_path) is False


def test_write_and_read_kill(tmp_path):
    state = write_control(tmp_path, kill_switch=True)
    assert state["kill_switch"] is True
    assert "updated_utc" in state
    assert kill_active(tmp_path) is True
    # flip back off
    write_control(tmp_path, kill_switch=False)
    assert kill_active(tmp_path) is False


def test_corrupt_file_falls_back(tmp_path):
    (tmp_path / "control.json").write_text("{not json", encoding="utf-8")
    assert read_control(tmp_path) == {"kill_switch": False}
