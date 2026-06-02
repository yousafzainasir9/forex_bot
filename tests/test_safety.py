"""Tests for the account-level safety helpers: circuit-breakers and the
correlation/exposure cap. All pure functions — no MT5, no network."""

from bot.risk import breaker_reason
from bot.scanner import currencies_of, exposure_breach


# --------------------------- circuit-breakers ----------------------------
def test_breaker_trips_on_consecutive_losses():
    assert breaker_reason(consecutive_losses=6, max_consecutive_losses=6,
                          drawdown_fraction=0.0, max_total_drawdown=0.0)
    assert breaker_reason(consecutive_losses=5, max_consecutive_losses=6,
                          drawdown_fraction=0.0, max_total_drawdown=0.0) is None


def test_breaker_trips_on_drawdown():
    assert breaker_reason(consecutive_losses=0, max_consecutive_losses=0,
                          drawdown_fraction=0.20, max_total_drawdown=0.15)
    assert breaker_reason(consecutive_losses=0, max_consecutive_losses=0,
                          drawdown_fraction=0.10, max_total_drawdown=0.15) is None


def test_breaker_disabled_with_zero_limits():
    # Both limits off -> never trips, no matter how bad the inputs look.
    assert breaker_reason(consecutive_losses=99, max_consecutive_losses=0,
                          drawdown_fraction=0.99, max_total_drawdown=0.0) is None


# --------------------------- exposure / correlation ----------------------
def test_currencies_of_decomposition():
    assert currencies_of("EURUSD") == {"EUR", "USD"}
    assert currencies_of("XAUUSD") == {"XAU", "USD"}
    assert currencies_of("BTCUSD") == {"BTC", "USD"}
    assert currencies_of("EURUSD.raw") == {"EUR", "USD"}
    assert currencies_of("US30") == {"US30"}     # index -> atomic
    assert currencies_of("NAS100") == {"NAS100"}  # has digits -> atomic


def test_exposure_breach_blocks_shared_currency():
    assert exposure_breach({"EURUSD"}, "EURGBP", 1)   # shares EUR
    assert exposure_breach({"EURUSD"}, "GBPUSD", 1)   # shares USD
    assert not exposure_breach({"EURUSD"}, "AUDJPY", 1)  # nothing shared


def test_exposure_breach_respects_cap_and_disable():
    assert not exposure_breach({"EURUSD"}, "EURGBP", 0)  # 0 disables the cap
    assert not exposure_breach({"EURUSD"}, "EURGBP", 2)  # cap 2 allows a 2nd EUR
    assert exposure_breach({"EURUSD", "EURJPY"}, "EURGBP", 2)  # 3rd EUR breaches cap 2
