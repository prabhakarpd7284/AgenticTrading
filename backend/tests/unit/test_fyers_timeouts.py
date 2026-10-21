"""Fyers SDK calls are bounded: a hung `funds` must not freeze the caller.

Regression: Fyers' funds endpoint hung indefinitely, freezing BrokerLink
snapshots, /brokers/{id}/diagnose/ and `fyers_login`.
"""
from __future__ import annotations

import threading
import time

import pytest

from plugins.broker_fyers import adapter as fy
from plugins.broker_fyers.adapter import FyersAdapter, call_with_timeout


def test_call_with_timeout_returns_result_and_passes_kwargs():
    assert call_with_timeout(lambda data: data["x"] * 2, 1, "probe", data={"x": 21}) == 42


def test_call_with_timeout_reraises_sdk_errors():
    def boom():
        raise ValueError("sdk said no")
    with pytest.raises(ValueError, match="sdk said no"):
        call_with_timeout(boom, 1, "probe")


def test_call_with_timeout_abandons_hung_call_quickly():
    release = threading.Event()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="Fyers probe timed out"):
        call_with_timeout(release.wait, 0.2, "probe")
    assert time.monotonic() - t0 < 1.0
    release.set()


class _HungFundsApi:
    def __init__(self):
        self.release = threading.Event()

    def funds(self):
        self.release.wait()
        return {"s": "ok", "fund_limit": []}


def test_fetch_margin_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(fy, "_FUNDS_TIMEOUT", 0.2)
    a = FyersAdapter(credentials={"app_id": "X-100", "access_token": "t"})
    a._api = _HungFundsApi()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="funds timed out"):
        a.fetch_margin()
    assert time.monotonic() - t0 < 1.0
    a._api.release.set()
