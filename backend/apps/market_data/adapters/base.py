"""Broker adapter base interface + common dataclasses.

Every broker adapter (angel_one, zerodha, fyers, paper) implements this
interface so the rest of the stack — the combined positions endpoint, the
Celery refresh task, the trade lifecycle service — can treat brokers
interchangeably.

Method results use a small set of broker-neutral dataclasses (`Position`,
`Holding`, `Margin`) so per-broker payload differences don't leak into
the rest of the codebase. Adapter implementations are responsible for
normalising their broker's response shape into these dataclasses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Position:
    """A currently-open intraday or carryforward position at a broker.

    `pnl` is realised+unrealised, `mtm` is mark-to-market unrealised only.
    Quantities are signed: positive long, negative short.
    """
    symbol: str
    exchange: str          # NSE | NFO | BSE | BFO | MCX
    product: str           # INTRADAY | DELIVERY | CARRYFORWARD
    quantity: int
    avg_price: float
    last_price: float
    pnl: float
    mtm: float = 0.0
    instrument_token: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Holding:
    """A delivery-segment holding (T+1 settled, sitting in the demat)."""
    symbol: str
    exchange: str
    quantity: int
    avg_price: float
    last_price: float
    pnl: float
    instrument_token: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Margin:
    """Capital snapshot at the broker."""
    available_cash: float
    used: float
    total: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerHealth:
    ok: bool
    detail: str = ""
    latency_ms: float = 0.0


# ── Options-chain dataclasses ─────────────────────────────────────────
# Used by every adapter's options_chain() implementation. The data model
# captures everything a strategy needs to size, price, and risk-check an
# options trade: bid/ask depth, OI, greeks, and source attribution so
# the strategy knows which broker quoted it.

@dataclass
class OptionQuote:
    """Per-leg quote with depth and greeks."""
    token: str                     # broker-specific instrument token
    symbol: str                    # e.g. NIFTY28APR2624050PE
    strike: int                    # in rupees (not paise)
    opt: str                       # "CE" | "PE"
    ltp: float
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    volume: int = 0
    oi: int = 0                    # open interest (lots × lot_size at this strike)
    oi_change: int = 0             # day's net change in OI
    # Greeks — empty (0.0) when the source doesn't provide them and we
    # haven't back-filled via BSM solver yet.
    iv: float = 0.0                # implied vol, decimal (0.18 = 18%)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0             # per-day, in price-points
    vega: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        """Mid-price — caller's safe default when bid/ask both present."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ltp

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points of mid. 0 if no quotes."""
        m = self.mid
        if m <= 0 or self.bid <= 0 or self.ask <= 0:
            return 0.0
        return (self.ask - self.bid) / m * 10_000


@dataclass
class OptionsChainRow:
    """One strike, both sides (when both exist)."""
    strike: int
    ce: Optional[OptionQuote] = None
    pe: Optional[OptionQuote] = None


@dataclass
class OptionsChainSnapshot:
    """A point-in-time options chain for one underlying + expiry."""
    underlying: str                # NIFTY | BANKNIFTY | SENSEX
    spot: float                    # current underlying LTP
    expiry: str                    # canonical "28APR2026" form
    fetched_at: datetime
    rows: list[OptionsChainRow]
    source: str                    # which adapter produced this
    # Optional underlying-level info — sources that have it (Fyers, Angel)
    # fill these; sources that don't (paper) leave them None.
    vix: Optional[float] = None
    pcr_oi: Optional[float] = None
    pcr_volume: Optional[float] = None
    atm_strike: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)

    def find(self, strike: int, opt: str) -> Optional[OptionQuote]:
        """Locate a single leg by (strike, opt). None if missing."""
        for r in self.rows:
            if r.strike == strike:
                return r.ce if opt == "CE" else r.pe
        return None

    def atm(self) -> Optional[int]:
        """Return ATM strike — uses cached field if set, else nearest to spot."""
        if self.atm_strike is not None:
            return self.atm_strike
        if not self.rows or self.spot <= 0:
            return None
        return min((r.strike for r in self.rows), key=lambda k: abs(k - self.spot))


class BrokerAdapterBase(ABC):
    """Adapter contract — every broker (angel_one, zerodha, fyers) implements this."""

    name: str = ""

    def __init__(self, credentials: dict[str, Any], meta: dict[str, Any] | None = None):
        """Credentials are the decrypted dict from BrokerLink.credential_blob.
        `meta` carries non-secret per-account info (account_id, user_code, etc.).
        """
        self.credentials = credentials
        self.meta = meta or {}

    # ── Authentication ────────────────────────────────────────────────
    @abstractmethod
    def authenticate(self) -> bool:
        """Establish a session / refresh token. Return True on success."""

    @abstractmethod
    def health_check(self) -> BrokerHealth:
        """Light ping used by /brokers/{id}/refresh/ to confirm connectivity."""

    # ── Read API ──────────────────────────────────────────────────────
    @abstractmethod
    def fetch_positions(self) -> list[Position]:
        """Open positions (intraday + carryforward)."""

    @abstractmethod
    def fetch_holdings(self) -> list[Holding]:
        """Delivery holdings sitting in the demat."""

    @abstractmethod
    def fetch_margin(self) -> Margin:
        """Current cash + margin utilisation."""

    # ── Options chain (optional — defined-risk strategies use this) ───
    def options_chain(
        self,
        underlying: str,
        expiry: Optional[str] = None,
        strikes_window: int = 20,
    ) -> Optional[OptionsChainSnapshot]:
        """Fetch a point-in-time options chain anchored at ATM.

        Args:
            underlying:    NIFTY | BANKNIFTY | SENSEX (per index conventions)
            expiry:        "28APR2026" — None picks the nearest expiry
            strikes_window: number of strikes either side of ATM to fetch.
                            Default 20 ⇒ 41 strikes total (20 below, ATM, 20 above).
                            Each strike has up to 2 legs (CE, PE).

        Returns:
            OptionsChainSnapshot with greeks where the source supports it,
            or None when chain fetch is unavailable (read-only adapter,
            credentials missing, etc.). Strategies treat None as a hard
            error and skip the cycle.
        """
        return None    # default: not supported. Live adapters override.

    # ── Write API (optional for read-only adapters) ───────────────────
    def place(self, order: dict) -> str:  # noqa: ARG002
        raise NotImplementedError(f"{self.name} adapter does not support place()")

    def cancel(self, order_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(f"{self.name} adapter does not support cancel()")

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def serialise_positions(items: list[Position]) -> list[dict]:
        return [_drop_raw(asdict(p)) for p in items]

    @staticmethod
    def serialise_holdings(items: list[Holding]) -> list[dict]:
        return [_drop_raw(asdict(h)) for h in items]

    @staticmethod
    def serialise_margin(m: Margin) -> dict:
        return _drop_raw(asdict(m))


def _drop_raw(d: dict) -> dict:
    """Strip the `raw` broker-specific payload before persisting/serialising."""
    return {k: v for k, v in d.items() if k != "raw"}
