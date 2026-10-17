"""CAMS / KFintech Consolidated Account Statement (CAS) -> plain dataclasses.

PDF reading is delegated to the ``casparser`` package (pypdfium2 backend);
this module only adapts its output into AlphaDesk's own small shape so the
importer (``cas_import``) and the tests never depend on casparser's pydantic
models. ``from_cas_data`` accepts a casparser ``CASData`` *or* the equivalent
plain dict, which is how the tests exercise everything without a PDF.

Scope: CAMS/KFin mutual-fund CAS (DETAILED gives transaction history; SUMMARY
gives closing balances only). NSDL/CDSL demat CAS files are not supported here
and are rejected with a clear error.

Secrets: the PDF password is PAN+DOB. It is never logged, and any text taken
from a library exception is scrubbed of it before being re-raised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# casparser TransactionType name -> MFTransaction.txn_type value.
# Anything not listed (STT/stamp-duty/TDS lines, segregation, gifts, MISC,
# UNKNOWN) is not a unit-moving buy/sell and is skipped, but counted.
TXN_TYPE_MAP = {
    "PURCHASE": "buy",
    "PURCHASE_SIP": "sip",
    "REDEMPTION": "sell",
    "SWITCH_IN": "switch_in",
    "SWITCH_IN_MERGER": "switch_in",
    "SWITCH_OUT": "switch_out",
    "SWITCH_OUT_MERGER": "switch_out",
    "DIVIDEND_PAYOUT": "dividend",
    "DIVIDEND_REINVEST": "dividend",
}

_ZERO = Decimal("0")


class CASParseError(Exception):
    """The statement could not be read (bad password, wrong format, corrupt)."""


@dataclass(frozen=True)
class ParsedTxn:
    txn_date: date
    txn_type: str  # buy | sell | sip | switch_in | switch_out | dividend
    units: Decimal  # absolute value; direction is carried by txn_type
    nav: Decimal
    amount: Decimal  # absolute value
    description: str = ""


@dataclass
class ParsedScheme:
    folio: str
    amc: str
    name: str
    isin: str | None
    amfi: str | None
    close_units: Decimal
    cost: Decimal | None  # total cost of the closing balance, if printed
    transactions: list[ParsedTxn] = field(default_factory=list)


@dataclass
class ParsedCAS:
    file_type: str  # CAMS | KFINTECH
    cas_type: str  # DETAILED | SUMMARY
    period_from: str
    period_to: str
    schemes: list[ParsedScheme]
    skipped_txn_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _enum_str(value: Any) -> str:
    return str(getattr(value, "value", value))


def _dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _to_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _as_dict(cas: Any) -> dict:
    return cas.model_dump(by_alias=True) if hasattr(cas, "model_dump") else cas


def from_cas_data(cas: Any) -> ParsedCAS:
    """Adapts casparser's ``CASData`` (or its dict form) into ``ParsedCAS``."""
    data = _as_dict(cas)
    if "folios" not in data:
        raise CASParseError(
            "This looks like a demat (NSDL/CDSL) statement. Only CAMS/KFintech "
            "mutual-fund CAS files are supported."
        )

    period = data.get("statement_period") or {}
    out = ParsedCAS(
        file_type=_enum_str(data.get("file_type", "UNKNOWN")),
        cas_type=_enum_str(data.get("cas_type", "UNKNOWN")),
        period_from=str(period.get("from", period.get("from_", ""))),
        period_to=str(period.get("to", "")),
        schemes=[],
        warnings=list(data.get("parse_warnings") or []),
    )

    for folio in data["folios"]:
        for sch in folio.get("schemes", []):
            close = _dec(sch.get("close"))
            if close is None:
                out.warnings.append(f"folio {folio.get('folio')}: scheme '{sch.get('scheme')}' has no closing balance; skipped")
                continue
            valuation = sch.get("valuation") or {}
            parsed = ParsedScheme(
                folio=str(folio.get("folio", "")).strip(),
                amc=str(folio.get("amc", "")).strip(),
                name=str(sch.get("scheme", "")).strip(),
                isin=(sch.get("isin") or None),
                amfi=(str(sch["amfi"]) if sch.get("amfi") else None),
                close_units=close,
                cost=_dec(valuation.get("cost")),
            )
            for t in sch.get("transactions", []):
                kind = TXN_TYPE_MAP.get(_enum_str(t.get("type")))
                d = _to_date(t.get("date"))
                if kind is None or d is None:
                    key = _enum_str(t.get("type")) if kind is None else "BAD_DATE"
                    out.skipped_txn_counts[key] = out.skipped_txn_counts.get(key, 0) + 1
                    continue
                parsed.transactions.append(
                    ParsedTxn(
                        txn_date=d,
                        txn_type=kind,
                        units=abs(_dec(t.get("units")) or _ZERO),
                        nav=abs(_dec(t.get("nav")) or _ZERO),
                        amount=abs(_dec(t.get("amount")) or _ZERO),
                        description=str(t.get("description", ""))[:200],
                    )
                )
            out.schemes.append(parsed)
    return out


def parse_cas_pdf(path: str, password: str) -> ParsedCAS:
    """Reads a CAS PDF. Raises ``CASParseError`` with a password-free message."""
    try:
        import casparser
        from casparser.exceptions import IncorrectPasswordError, ParserException
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CASParseError("casparser is not installed (uv pip install casparser)") from exc

    def scrub(text: str) -> str:
        return text.replace(password, "***") if password else text

    try:
        cas = casparser.read_cas_pdf(path, password)
    except IncorrectPasswordError:
        raise CASParseError("Incorrect PDF password (expected PAN+DOB as printed in the email).") from None
    except FileNotFoundError:
        raise CASParseError(f"File not found: {path}") from None
    except ParserException as exc:
        raise CASParseError(f"Could not parse statement: {scrub(str(exc))}") from None
    except Exception as exc:  # corrupt/unsupported PDF from the PDF backend
        raise CASParseError(f"Could not read PDF ({type(exc).__name__}): {scrub(str(exc))}") from None

    parsed = from_cas_data(cas)
    log.info(
        "cas_parse.ok",
        file_type=parsed.file_type,
        cas_type=parsed.cas_type,
        schemes=len(parsed.schemes),
        txns=sum(len(s.transactions) for s in parsed.schemes),
        warnings=len(parsed.warnings),
    )
    return parsed
