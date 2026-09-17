"""Group tokens by exchange for the SmartWebSocketV2 subscribe payload.

The screener's tick stream hardcoded ``{"exchangeType": 1, ...}`` — NSE cash.
Subscribing an NFO option token under exchangeType 1 does not error; the feed
connects and simply never delivers that token. So the socket looks healthy
while carrying nothing, which is this system's signature failure mode.
"""
from __future__ import annotations

import pytest

from apps.market_data.services.ws_subscription import (
    EXCHANGE_TYPE,
    exchange_type,
    subscription_batches,
)


def test_known_exchanges_map_to_their_v2_codes():
    assert exchange_type("NSE") == 1     # nse_cm
    assert exchange_type("NFO") == 2     # nse_fo
    assert exchange_type("BSE") == 3     # bse_cm
    assert exchange_type("BFO") == 4     # bse_fo


def test_lookup_is_case_insensitive():
    assert exchange_type("nfo") == EXCHANGE_TYPE["NFO"]


def test_unknown_exchange_is_rejected_rather_than_defaulted():
    """Defaulting to NSE is how option tokens end up silently unsubscribed."""
    with pytest.raises(ValueError):
        exchange_type("XYZ")


def test_tokens_are_grouped_by_exchange():
    batches = subscription_batches({
        "2885": "NSE", "1594": "NSE", "47297": "NFO", "47298": "NFO",
    })

    by_type = {b["exchangeType"]: set(b["tokens"]) for b in batches}
    assert by_type[1] == {"2885", "1594"}
    assert by_type[2] == {"47297", "47298"}


def test_each_exchange_is_batched_at_the_subscribe_limit():
    tokens = {str(i): "NSE" for i in range(120)}

    batches = subscription_batches(tokens, batch_size=50)

    assert [len(b["tokens"]) for b in batches] == [50, 50, 20]
    assert all(b["exchangeType"] == 1 for b in batches)


def test_mixed_universe_batches_each_exchange_separately():
    tokens = {str(i): "NSE" for i in range(60)}
    tokens.update({f"opt{i}": "NFO" for i in range(10)})

    batches = subscription_batches(tokens, batch_size=50)

    nse = [b for b in batches if b["exchangeType"] == 1]
    nfo = [b for b in batches if b["exchangeType"] == 2]
    assert [len(b["tokens"]) for b in nse] == [50, 10]
    assert [len(b["tokens"]) for b in nfo] == [10]


def test_unknown_exchange_is_skipped_not_mislabelled():
    """One bad row must not poison the whole subscription."""
    batches = subscription_batches({"2885": "NSE", "999": "MCX_WEIRD"})

    all_tokens = {t for b in batches for t in b["tokens"]}
    assert all_tokens == {"2885"}


def test_no_tokens_yields_no_batches():
    assert subscription_batches({}) == []
