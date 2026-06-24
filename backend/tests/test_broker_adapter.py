"""Regression tests for the Angel One adapter.

Most of the suite MOCKS the broker adapter, so a missing import or undefined
name in the real adapter slips past every other test and only explodes at
runtime against a live broker link (which is exactly how
`NameError: name 'time' is not defined` reached production on 2026-06-24 after
the rate-limit breaker rewrite dropped `import time`). These tests exercise the
real adapter code paths without any network.
"""
import time

import pytest


def test_ensure_exercises_time_without_nameerror():
    """_ensure()/authenticate()/health_check() all call time.time(); guard the
    import so a future refactor can't silently drop it again."""
    from plugins.broker_angel.adapter import AngelOneAdapter

    a = AngelOneAdapter({
        "api_key": "k", "client_code": "c", "password": "p", "totp_secret": "x",
    })
    a._api = object()                       # pretend already authenticated
    a._session_expires = time.time() + 3600
    # Would raise NameError if `import time` were missing again.
    assert a._ensure() is True


def test_throttled_short_circuits_while_breaker_open():
    """When the shared breaker is open, _throttled must fast-fail with
    BrokerRateLimited and never invoke the wrapped call (no network hit)."""
    import plugins.broker_angel.adapter as adapter
    from trading.services.data_service import BrokerClient

    bc = BrokerClient.get_instance()
    bc.reset_breaker()
    try:
        bc.trip_breaker()
        called = {"n": 0}
        with pytest.raises(adapter.BrokerRateLimited):
            adapter._throttled(lambda: called.__setitem__("n", 1))
        assert called["n"] == 0
    finally:
        bc.reset_breaker()


def test_breaker_half_open_allows_single_probe_then_closes():
    """After the cooldown elapses the breaker is HALF-OPEN: exactly one caller
    gets 'probe', the rest get 'open', and a successful probe closes it — so
    traffic isn't starved for the whole window and never re-storms."""
    from trading.services.data_service import BrokerClient, _get_redis_for_throttle

    bc = BrokerClient.get_instance()
    bc.reset_breaker()
    r = _get_redis_for_throttle()
    if r is None:
        return  # Redis-less env: half-open coordination isn't exercised
    try:
        bc.trip_breaker()
        assert bc.breaker_gate() == "open"
        # Simulate the cooldown having elapsed without sleeping.
        import time as _t
        r.set(bc._REDIS_BREAKER_KEY, int(_t.time() * 1000) - 1000)
        r.delete(bc._REDIS_BREAKER_PROBE_KEY)
        assert bc.breaker_gate() == "probe"   # first caller wins the probe
        assert bc.breaker_gate() == "open"    # probe already taken → others wait
        bc.reset_breaker()                    # probe succeeded → resume traffic
        assert bc.breaker_gate() == "closed"
    finally:
        bc.reset_breaker()
