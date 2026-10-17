"""Upserts a parsed CAS into MFHolding + MFTransaction (source='cas').

Matching a CAS scheme to the AMFI catalog (MFScheme), in order:

1. **ISIN** (``isin_growth`` / ``isin_div_reinvestment``) — authoritative.
2. **AMFI scheme code** printed in the statement, when the ISIN is absent or
   not in the catalog (``MFScheme.scheme_code`` *is* the AMFI code).
3. **AMC + normalized scheme name** — only when the statement has *no* ISIN,
   and only on an exact, unambiguous normalized match.

Anything that does not match, or matches ambiguously, is skipped and reported
in ``ImportResult.unmatched`` — a bad row never aborts the run.

Idempotent: re-importing the same statement creates no new rows. Transactions
have no natural key, so they are compared as a multiset on
(type, date, units, nav, amount) per holding; identical same-day transactions
are therefore preserved, not collapsed.

Units/amount are stored as absolute values — direction lives in ``txn_type``.
Importing over an existing *manual* holding for the same
(portfolio, scheme, folio) takes it over (source -> 'cas', units from the CAS).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import structlog
from django.db import transaction
from django.db.models import Q

from apps.mutual_funds.models import MFHolding, MFScheme, MFTransaction
from apps.mutual_funds.services.cas_parser import ParsedCAS, ParsedScheme

log = structlog.get_logger(__name__)

_Q4 = Decimal("0.0001")
_Q2 = Decimal("0.01")


@dataclass
class ImportResult:
    schemes_total: int = 0
    matched: int = 0
    unmatched: list[dict] = field(default_factory=list)
    holdings_created: int = 0
    holdings_updated: int = 0
    txns_created: int = 0
    txns_existing: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


# ── scheme matching ────────────────────────────────────────────────────

_RTA_PREFIX = re.compile(r"^\s*[A-Z0-9]{2,8}\s*-\s*(?=[A-Za-z])")
_PARENS = re.compile(r"\([^)]*\)")
_ISIN_TAIL = re.compile(r"\bisin\b.*$", re.IGNORECASE)
_FILLER_WORDS = {"plan", "option"}


def normalize_name(name: str) -> str:
    """Lowercased alphanumerics of a scheme name with the statement's
    decorations (RTA code prefix, parentheticals, trailing ISIN) removed.
    The filler words "plan"/"option" are dropped because CAS and AMFI differ
    on them ("Direct Plan - Growth" vs "Direct Plan - Growth Option" vs
    "Direct Growth"); "direct"/"regular"/"growth"/"idcw" are kept, so plan and
    option still distinguish schemes."""
    s = _RTA_PREFIX.sub("", name or "")
    s = _ISIN_TAIL.sub("", s)
    s = _PARENS.sub(" ", s)
    words = re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
    return " ".join(w for w in words if w not in _FILLER_WORDS)


def _amc_key(amc: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", amc.lower().replace("mutual fund", "")).strip()


def match_scheme(ps: ParsedScheme) -> tuple[MFScheme | None, str]:
    """Returns ``(scheme, how)`` on success or ``(None, reason)``."""
    if ps.isin:
        hits = list(MFScheme.objects.filter(Q(isin_growth=ps.isin) | Q(isin_div_reinvestment=ps.isin))[:3])
        if len(hits) == 1:
            return hits[0], "isin"
        if len(hits) > 1:
            by_code = [h for h in hits if ps.amfi and h.scheme_code == ps.amfi]
            if len(by_code) == 1:
                return by_code[0], "isin+amfi"
            return None, "ambiguous_isin"

    if ps.amfi:
        hit = MFScheme.objects.filter(pk=ps.amfi).first()
        if hit:
            return hit, "amfi_code"

    if ps.isin:
        # An ISIN was printed but the catalog has no such scheme (merged /
        # renamed / wound up). Do NOT fuzzy-match around an explicit ISIN.
        return None, "isin_not_in_catalog"

    target = normalize_name(ps.name)
    amc = _amc_key(ps.amc)
    if not target or not amc:
        return None, "no_identifiers"
    first = amc.split(" ")[0]
    candidates = MFScheme.objects.filter(amc_name__istartswith=first)
    hits = [s for s in candidates if _amc_key(s.amc_name) == amc and normalize_name(s.scheme_name) == target]
    if len(hits) == 1:
        return hits[0], "name"
    return None, "ambiguous_name" if hits else "name_not_found"


# ── import ─────────────────────────────────────────────────────────────

def _mask(folio: str) -> str:
    return f"…{folio[-4:]}" if len(folio) > 4 else "…"


def _txn_key(t) -> tuple:
    return (
        t.txn_type,
        t.txn_date,
        Decimal(t.units).quantize(_Q4, ROUND_HALF_UP),
        Decimal(t.nav).quantize(_Q4, ROUND_HALF_UP),
        Decimal(t.amount).quantize(_Q2, ROUND_HALF_UP),
    )


def _import_scheme(ps, scheme, *, tenant, portfolio, dry_run, res: ImportResult) -> None:
    close = ps.close_units.quantize(_Q4, ROUND_HALF_UP)
    avg_cost = None
    if ps.cost is not None and close > 0:
        avg_cost = (ps.cost / close).quantize(_Q4, ROUND_HALF_UP)

    holding = MFHolding.objects.filter(
        tenant=tenant, portfolio=portfolio, scheme=scheme, folio_number=ps.folio
    ).first()

    if dry_run:
        res.holdings_updated += 1 if holding else 0
        res.holdings_created += 0 if holding else 1
        existing = Counter()
        if holding:
            existing = Counter(_txn_key(t) for t in holding.transactions.all())
    else:
        if holding is None:
            holding = MFHolding.objects.create(
                tenant=tenant, portfolio=portfolio, scheme=scheme, folio_number=ps.folio,
                units=close, avg_cost_per_unit=avg_cost, source=MFHolding.SOURCE_CAS,
            )
            res.holdings_created += 1
        else:
            holding.units = close
            holding.avg_cost_per_unit = avg_cost
            holding.source = MFHolding.SOURCE_CAS
            holding.save(update_fields=["units", "avg_cost_per_unit", "source", "updated_at"])
            res.holdings_updated += 1
        existing = Counter(_txn_key(t) for t in holding.transactions.all())

    wanted = Counter(_txn_key(t) for t in ps.transactions)
    to_create: list[MFTransaction] = []
    for key, n_stmt in wanted.items():
        missing = n_stmt - existing.get(key, 0)
        res.txns_existing += min(n_stmt, existing.get(key, 0))
        res.txns_created += max(missing, 0)
        if missing > 0 and not dry_run:
            ttype, tdate, units, nav, amount = key
            to_create += [
                MFTransaction(
                    tenant=tenant, holding=holding, txn_type=ttype, txn_date=tdate,
                    units=units, nav=nav, amount=amount,
                )
                for _ in range(missing)
            ]
    if to_create:
        MFTransaction.objects.bulk_create(to_create, batch_size=1000)


def import_cas(parsed: ParsedCAS, *, tenant, portfolio, dry_run: bool = False) -> ImportResult:
    """Imports every scheme in ``parsed``. Per-scheme failures are isolated in
    a savepoint and recorded in ``result.errors``; the run continues."""
    if portfolio.tenant_id != tenant.id:
        raise ValueError("portfolio does not belong to tenant")

    res = ImportResult(schemes_total=len(parsed.schemes), dry_run=dry_run)
    for ps in parsed.schemes:
        scheme, how = match_scheme(ps)
        if scheme is None:
            res.unmatched.append({"scheme": ps.name, "isin": ps.isin, "amc": ps.amc, "reason": how})
            log.warning("cas_import.unmatched", scheme=ps.name, isin=ps.isin, reason=how)
            continue
        part = ImportResult()  # counted apart so a rolled-back scheme leaves no trace in res
        try:
            with transaction.atomic():
                _import_scheme(ps, scheme, tenant=tenant, portfolio=portfolio, dry_run=dry_run, res=part)
            res.matched += 1
            res.holdings_created += part.holdings_created
            res.holdings_updated += part.holdings_updated
            res.txns_created += part.txns_created
            res.txns_existing += part.txns_existing
        except Exception as exc:  # noqa: BLE001 - one bad scheme must not stop the run
            res.errors.append(f"{ps.name} (folio {_mask(ps.folio)}): {type(exc).__name__}: {exc}")
            log.error("cas_import.scheme_failed", scheme=ps.name, folio=_mask(ps.folio), error=str(exc))

    log.info(
        "cas_import.complete",
        dry_run=dry_run, schemes=res.schemes_total, matched=res.matched,
        unmatched=len(res.unmatched), holdings_created=res.holdings_created,
        holdings_updated=res.holdings_updated, txns_created=res.txns_created,
        txns_existing=res.txns_existing, errors=len(res.errors),
    )
    return res
