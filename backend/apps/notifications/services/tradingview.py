"""TradingView webhook payload pipeline.

TradingView Pro+ lets users paste a webhook URL into any alert. When the
alert fires, TradingView POSTs the rendered message body to that URL. There
is no authentication header — the only thing proving the request came from
the right account is the unguessable secret in the URL path.

This module is the *receive* side: parse the payload, record it for the
audit trail, persist as a v2 Signal so it shows up on the Now feed + the
monthly capture-rate report, and (when the operator opted in) spawn an
AgentRun for the configured strategy.

Three concerns are deliberately kept separate so the webhook view stays
thin: parsing, persistence, and workflow dispatch.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from apps.notifications.models import TradingViewLink, TradingViewSignal

log = structlog.get_logger()


# ── Parsing ──────────────────────────────────────────────────────────────

@dataclass
class ParsedAlert:
    """Normalised view of a TradingView alert payload.

    `action` is upper-cased and trimmed; downstream code can rely on
    "BUY"/"SELL"/"EXIT" etc. without re-normalising. `price` is parsed to
    float when present; everything else preserved as-is in `extra`."""
    symbol: str = ""
    action: str = ""
    price: float | None = None
    strategy: str = ""
    comment: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol":   self.symbol,
            "action":   self.action,
            "price":    self.price,
            "strategy": self.strategy,
            "comment":  self.comment,
        }
        if self.extra:
            out["extra"] = self.extra
        return out


# `BUY RELIANCE @ 1234.56` / `SELL TCS 3500` / `BUY HDFCBANK`
_PLAIN_RE = re.compile(
    r"""^\s*
        (?P<action>BUY|SELL|EXIT|LONG|SHORT|CLOSE)\s+
        (?P<symbol>[A-Z][A-Z0-9_-]{0,29})
        (?:\s+(?:@\s*)?(?P<price>\d+(?:\.\d+)?))?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_payload(body: str, content_type: str = "") -> ParsedAlert:
    """Best-effort parse of a TradingView webhook body.

    Tries JSON first (the convention TradingView power-users settle on),
    falls back to a `<ACTION> <SYMBOL> [@ <PRICE>]` plaintext regex. Raises
    ValueError when both paths fail so the caller can record the raw payload
    + parse error for operator inspection.
    """
    body = (body or "").strip()
    if not body:
        raise ValueError("empty body")

    # JSON path: accept both `application/json` and the much-more-common
    # case where TradingView posts a JSON string body with text/plain
    # content-type. Try parse-as-JSON unconditionally; only fall back to
    # plaintext if the body doesn't look like JSON.
    if body.startswith("{") or "json" in content_type.lower():
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON payload must be an object")
        return _from_json(data)

    # Plaintext path
    m = _PLAIN_RE.match(body)
    if not m:
        raise ValueError(
            "could not parse plaintext payload — expected '<BUY|SELL> <SYMBOL> [@ <PRICE>]' "
            f"or JSON; got {body[:80]!r}"
        )
    return ParsedAlert(
        symbol=m.group("symbol").upper(),
        action=m.group("action").upper(),
        price=float(m.group("price")) if m.group("price") else None,
        strategy="",
        comment=body,
    )


def _from_json(data: dict[str, Any]) -> ParsedAlert:
    # TradingView users frequently use lowercase OR snake_case OR all-caps
    # in their alert template. Normalise.
    def pick(*names: str, default: Any = "") -> Any:
        for n in names:
            for variant in (n, n.lower(), n.upper(), n.replace("_", "")):
                if variant in data:
                    return data[variant]
        return default

    symbol = str(pick("symbol", "ticker", default="") or "").upper().strip()
    action = str(pick("action", "side", "order_action", default="") or "").upper().strip()
    raw_price = pick("price", "close", default=None)
    try:
        price = float(raw_price) if raw_price not in (None, "") else None
    except (TypeError, ValueError):
        price = None

    return ParsedAlert(
        symbol=symbol,
        action=action,
        price=price,
        strategy=str(pick("strategy", "strategy_name", default="") or ""),
        comment=str(pick("comment", "message", default="") or ""),
        extra={k: v for k, v in data.items()
               if k.lower() not in {"symbol", "ticker", "action", "side", "order_action",
                                    "price", "close", "strategy", "strategy_name",
                                    "comment", "message"}},
    )


# ── Persistence ──────────────────────────────────────────────────────────

@transaction.atomic
def record_alert(
    link: TradingViewLink,
    raw_body: str,
    parsed: ParsedAlert | None,
    parse_error: str = "",
) -> TradingViewSignal:
    """Persist the webhook hit as a TradingViewSignal row.

    Also bumps `link.receive_count` and `link.last_received_at`. When parsing
    succeeded AND a symbol+action were extracted, also writes a row to
    `strategies.Signal` so the Now feed + monthly report pick it up.

    Errors during the strategies.Signal write are swallowed — the audit row
    on TradingViewSignal is the canonical record; downstream persistence is
    best-effort (matches Invariant 6 in CLAUDE.md).
    """
    from apps.strategies.models import Signal as StrategiesSignal

    parsed_payload = parsed.to_payload() if parsed else {}

    tv_signal = TradingViewSignal.objects.create(
        tenant=link.tenant,
        link=link,
        raw_payload=raw_body,
        parsed=parsed_payload,
        parse_error=parse_error or "",
    )

    if parsed and parsed.symbol and parsed.action:
        try:
            ledger = StrategiesSignal.objects.create(
                tenant=link.tenant,
                symbol=parsed.symbol,
                signal_date=timezone.now().date(),
                signal_time=timezone.now(),
                source=StrategiesSignal.Source.TRADINGVIEW,
                strategy=parsed.strategy or link.display_name or "tradingview",
                side=_normalise_side(parsed.action),
                entry_price=parsed.price or 0.0,
                stoploss=0.0,
                target=0.0,
                confidence=0.0,
                reasons=[parsed.comment] if parsed.comment else [],
                indicators={"tradingview_link": str(link.id), **(parsed.extra or {})},
            )
            tv_signal.signal = ledger
            tv_signal.save(update_fields=["signal"])
        except Exception:  # noqa: BLE001
            log.exception("tradingview.signal_persist_failed", link_id=str(link.id))

    # Bookkeeping on the link itself — non-blocking conceptually but inside
    # the atomic block so the counter never lies.
    TradingViewLink.objects.filter(pk=link.pk).update(
        last_received_at=timezone.now(),
        receive_count=F("receive_count") + 1,
        last_error=parse_error[:512] if parse_error else "",
    )

    return tv_signal


def _normalise_side(action: str) -> str:
    """Map TradingView action vocabulary onto the Signal.side BUY|SELL field.

    LONG/COVER → BUY, SHORT/EXIT/CLOSE → SELL. Unknown actions fall through
    as-is (Signal.side is a free CharField; the monthly report tolerates it)."""
    a = action.upper()
    if a in {"BUY", "LONG", "COVER"}: return "BUY"
    if a in {"SELL", "SHORT", "EXIT", "CLOSE"}: return "SELL"
    return a[:5]  # Signal.side is CharField(max_length=5)


# ── Auto-fire ────────────────────────────────────────────────────────────

def fire_workflow(link: TradingViewLink, parsed: ParsedAlert) -> str | None:
    """When autofire is enabled and the alert passes the allowlist gate,
    enqueue an AgentRun for `link.default_strategy_name`. Returns the run
    UUID as a string, or None if autofire was disabled / gated / failed.

    The actual trade still goes through the canonical 10-criterion RiskGuard
    inside the workflow — autofire only removes the manual click, never the
    risk check.
    """
    if not link.autofire_enabled:
        return None
    if not link.default_strategy_name:
        log.warning("tradingview.autofire_skipped", reason="no_default_strategy",
                    link_id=str(link.id))
        return None
    if link.portfolio_id is None:
        log.warning("tradingview.autofire_skipped", reason="no_portfolio",
                    link_id=str(link.id))
        return None
    allowed = link.allowed_actions or []
    if allowed and parsed.action not in allowed:
        log.info("tradingview.autofire_gated", reason="action_not_allowed",
                 link_id=str(link.id), action=parsed.action, allowed=allowed)
        return None

    # Symbol allowlist via bound watchlist. The watchlist's `symbols` field
    # is always populated (manual = operator-typed; auto = resolver-cached),
    # so this check is a single set membership — no extra queries.
    if link.watchlist_id is not None:
        wl_symbols = set(link.watchlist.symbols or [])
        if parsed.symbol not in wl_symbols:
            log.info("tradingview.autofire_gated", reason="symbol_not_in_watchlist",
                     link_id=str(link.id), symbol=parsed.symbol,
                     watchlist_id=str(link.watchlist_id))
            return None

    # Lazy import — avoids pulling celery + strategy_registry at module load,
    # which matters when the webhook view imports this services module.
    from apps.agents_core.models import AgentRun
    from apps.agents_core.registry import strategy_registry
    from apps.agents_core.tasks.run import execute_run

    try:
        strat = strategy_registry.get(link.default_strategy_name)
    except KeyError:
        log.error("tradingview.autofire_failed", reason="unknown_strategy",
                  link_id=str(link.id), strategy=link.default_strategy_name)
        return None

    config = {
        "symbol":   parsed.symbol,
        "action":   parsed.action,
        "price":    parsed.price,
        "comment":  parsed.comment,
        "origin":   "tradingview",
        **(parsed.extra or {}),
    }

    run = AgentRun.objects.create(
        tenant=link.tenant,
        triggered_by=link.owner,
        strategy_name=strat.name,
        strategy_version=strat.version,
        portfolio_id=link.portfolio_id,
        config=config,
    )
    execute_run.delay(str(run.id))
    return str(run.id)
