"""Build SmartWebSocketV2 subscription payloads for a mixed-exchange universe.

The screener's tick stream hardcoded ``{"exchangeType": 1}`` — NSE cash — so it
could only ever stream equities. Subscribing an NFO option token under
exchangeType 1 does not raise: the socket connects, reports itself healthy, and
simply never delivers that token. Same failure signature as everything else in
this system, so the unknown-exchange case here raises rather than defaults.

Angel's exchangeType codes are a fixed protocol enum, not configuration.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

EXCHANGE_TYPE = {
    "NSE": 1,    # nse_cm — equity cash
    "NFO": 2,    # nse_fo — NSE derivatives (NIFTY/BANKNIFTY options)
    "BSE": 3,    # bse_cm
    "BFO": 4,    # bse_fo — BSE derivatives (SENSEX options)
    "MCX": 5,    # mcx_fo
}

# Angel caps tokens per subscribe call; larger lists are silently truncated.
DEFAULT_BATCH_SIZE = 50


def exchange_type(exchange: str) -> int:
    """V2 exchangeType code for an exchange segment."""
    code = EXCHANGE_TYPE.get((exchange or "").upper())
    if code is None:
        raise ValueError(
            f"Unknown exchange {exchange!r}. Defaulting to NSE is how option "
            f"tokens end up subscribed to a segment that never delivers them."
        )
    return code


def subscription_batches(
    token_exchanges: dict[str, str], batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """``{token: exchange}`` → the token_list payload V2 expects.

    Grouped by exchange and chunked to the subscribe limit. A token whose
    exchange cannot be resolved is skipped with a warning rather than
    mislabelled onto a segment where it would never tick.
    """
    grouped: dict[int, list[str]] = {}
    for token, exchange in token_exchanges.items():
        try:
            code = exchange_type(exchange)
        except ValueError:
            log.warning(
                "ws_subscription.unknown_exchange", token=token, exchange=exchange,
            )
            continue
        grouped.setdefault(code, []).append(token)

    batches: list[dict] = []
    for code, tokens in sorted(grouped.items()):
        for i in range(0, len(tokens), batch_size):
            batches.append({
                "exchangeType": code,
                "tokens": tokens[i:i + batch_size],
            })
    return batches
