"""MF advisor — deterministic, rule-based findings over a tenant's MF holdings.

PERSONAL USE ONLY. This runs on the owner's own money and surfaces
arithmetic and data-quality observations. It is not investment advice to any
third party; giving personalised recommendations to anyone else would raise
the AMFI ARN / SEBI RIA registration question (see CLAUDE.md, Phase MF-3).
No LLM is involved and nothing here makes a market call.

Deliberately separate from the collection code (amfi_nav_collector,
amfi_history, staleness): this module only READS MFScheme / MFNav /
MFHolding / MFTransaction and never writes to them.

Four checks, each returning ``Finding`` rows:

1. ``direct_vs_regular``  — regular-plan holdings, their direct twin, and an
   ESTIMATED annual cost drag. AMFI's NAV file carries no expense ratios, so
   the drag is ``current_value x assumed_drag_pct`` where the percentage is an
   assumption supplied by the caller, never a fact. Where both plans have NAV
   history the observed growth-option return gap is reported alongside it.
2. ``allocation``         — current category mix vs a target the owner states
   (percent per category). Pure arithmetic on units x latest NAV.
3. ``overlap``            — category+AMC level only (same scheme held in two
   plans, or several schemes of one AMC in one category). Stock-level overlap
   needs each fund's underlying holdings, which we do not have: reported as an
   explicit open question, never faked.
4. ``sip_continuity``     — SIP cadence inferred from MFTransaction history;
   flags holdings whose latest SIP is overdue ("possibly lapsed").

Every finding is logged through ``event_writer.emit()`` (non-blocking). The
Event type enum has no advisor-specific member, so findings use the existing
``workflow.step.completed`` type with ``step_name="mf_advisor.<kind>"``.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from statistics import median

import structlog

from apps.mutual_funds.models import MFHolding, MFNav, MFScheme, MFTransaction

log = structlog.get_logger(__name__)

DEFAULT_ASSUMED_DRAG_PCT = 0.75   # assumed regular-minus-direct spread, % per year
DEFAULT_DRIFT_THRESHOLD_PCT = 5.0  # allocation drift (percentage points) worth flagging
SIP_MIN_HISTORY = 3                # SIP transactions needed to infer a cadence
SIP_GRACE_DAYS = 5
MIN_HISTORY_OVERLAP_DAYS = 180     # NAV overlap needed to quote an observed gap

OVERLAP_OPEN_QUESTION = (
    "Stock-level overlap is NOT computed: it needs each fund's underlying "
    "holdings, which AMFI's NAV file does not provide (needs factsheet "
    "scraping or a paid data vendor). Only category+AMC overlap is reported."
)

_PLAN_OPTION_TOKENS = {
    "direct", "regular", "plan", "growth", "option", "idcw", "dividend",
    "payout", "reinvestment", "reinvest", "income", "distribution", "cum",
    "capital", "withdrawal", "bonus", "div",
}
_IDCW_MARKERS = ("idcw", "dividend", "income distribution", "payout", "reinvest")


# ── value types ──────────────────────────────────────────────────────────

@dataclass
class Finding:
    kind: str                 # direct_vs_regular | allocation | overlap | sip_continuity | note
    severity: str             # info | warn
    text: str
    payload: dict = field(default_factory=dict)


@dataclass
class AdvisorReport:
    findings: list[Finding] = field(default_factory=list)
    unvalued: list[str] = field(default_factory=list)   # holdings with no NAV
    total_value: float = 0.0
    notes: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]


# ── helpers ──────────────────────────────────────────────────────────────

def normalize_scheme_name(name: str) -> str:
    """Scheme name with plan/option words stripped, for matching a regular
    plan to its direct twin ("X Fund - Direct Plan - Growth" == "X Fund")."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return " ".join(w for w in words if w not in _PLAN_OPTION_TOKENS)


def option_key(scheme: MFScheme) -> str:
    """'growth', or the normalised IDCW/dividend option text (so a monthly
    IDCW never matches a quarterly one)."""
    text = f"{scheme.option} {scheme.scheme_name}".lower()
    if "growth" in text and not any(m in scheme.option.lower() for m in _IDCW_MARKERS):
        return "growth"
    if any(m in text for m in _IDCW_MARKERS):
        opt = re.sub(r"[^a-z0-9]+", " ", scheme.option.lower())
        opt = " ".join(w for w in opt.split() if w not in {"plan", "option", "direct", "regular"})
        return f"idcw:{opt}"
    return "other"


def category_label(scheme: MFScheme) -> str:
    """'Open Ended Schemes(Equity Scheme - Large Cap Fund)' -> 'Equity Scheme - Large Cap Fund'."""
    m = re.search(r"\((.*)\)\s*$", scheme.category or "")
    label = m.group(1) if m else (scheme.category or "")
    return re.sub(r"\s+", " ", label).strip() or "Uncategorised"


def latest_nav(scheme_id: str) -> MFNav | None:
    return MFNav.objects.filter(scheme_id=scheme_id).order_by("-nav_date").first()


def _f(x) -> float:
    return float(x) if isinstance(x, Decimal) else x


def _valued(holdings) -> tuple[list[tuple[MFHolding, MFNav, float]], list[MFHolding]]:
    valued, unvalued = [], []
    for h in holdings:
        nav = latest_nav(h.scheme_id)
        if nav is None:
            unvalued.append(h)
        else:
            valued.append((h, nav, float(h.units * nav.nav)))
    return valued, unvalued


# ── 1. direct vs regular ─────────────────────────────────────────────────

def find_direct_twin(scheme: MFScheme) -> MFScheme | None:
    """Direct-plan scheme with the same AMC, normalised name and option."""
    key = normalize_scheme_name(scheme.scheme_name)
    opt = option_key(scheme)
    for cand in MFScheme.objects.filter(amc_name=scheme.amc_name, is_direct=True):
        if normalize_scheme_name(cand.scheme_name) == key and option_key(cand) == opt:
            return cand
    return None


def observed_gap_pct_pa(regular: MFScheme, direct: MFScheme) -> float | None:
    """Annualised growth-return gap (direct minus regular) over the common NAV
    window, or None if there is too little overlap / the option is not growth.
    Only meaningful for growth options (IDCW payouts distort NAV)."""
    if option_key(regular) != "growth":
        return None
    reg = {n.nav_date: n.nav for n in MFNav.objects.filter(scheme=regular)}
    dr = {n.nav_date: n.nav for n in MFNav.objects.filter(scheme=direct)}
    common = sorted(set(reg) & set(dr))
    if len(common) < 2:
        return None
    d0, d1 = common[0], common[-1]
    days = (d1 - d0).days
    if days < MIN_HISTORY_OVERLAP_DAYS or not reg[d0] or not dr[d0]:
        return None
    reg_ret = float(reg[d1] / reg[d0])
    dir_ret = float(dr[d1] / dr[d0])
    years = days / 365.25
    return round(((dir_ret ** (1 / years)) - (reg_ret ** (1 / years))) * 100, 2)


def check_direct_vs_regular(valued, assumed_drag_pct: float) -> list[Finding]:
    out = []
    for h, nav, value in valued:
        s = h.scheme
        if s.is_direct:
            continue
        twin = find_direct_twin(s)
        drag = round(value * assumed_drag_pct / 100, 2)
        payload = {
            "holding_id": h.id, "folio": h.folio_number,
            "regular_scheme_code": s.scheme_code, "regular_scheme_name": s.scheme_name,
            "direct_scheme_code": twin.scheme_code if twin else None,
            "current_value": round(value, 2),
            "assumed_drag_pct": assumed_drag_pct,
            "estimated_annual_drag_inr": drag,
            "is_estimate": True,
        }
        if twin:
            gap = observed_gap_pct_pa(s, twin)
            payload["observed_growth_gap_pct_pa"] = gap
            text = (
                f"{s.scheme_name} is a regular plan; direct twin {twin.scheme_code} exists. "
                f"ESTIMATED drag ~INR {drag:,.0f}/yr at an ASSUMED {assumed_drag_pct}% spread "
                f"(AMFI data has no expense ratios)"
                + (f"; observed growth gap over NAV history: {gap}%/yr" if gap is not None else "")
                + "."
            )
        else:
            text = (
                f"{s.scheme_name} is a regular plan; no direct twin found by AMC+name+option match. "
                f"ESTIMATED drag ~INR {drag:,.0f}/yr at an ASSUMED {assumed_drag_pct}% spread."
            )
        out.append(Finding("direct_vs_regular", "warn", text, payload))
    return out


# ── 2. allocation vs target ──────────────────────────────────────────────

def check_allocation(valued, target: dict[str, float], drift_threshold_pct: float) -> list[Finding]:
    total = sum(v for _, _, v in valued)
    if total <= 0:
        return []
    actual: dict[str, float] = defaultdict(float)
    for h, _, v in valued:
        actual[category_label(h.scheme)] += v
    tgt = {k.strip(): float(v) for k, v in target.items()}
    lower_tgt = {k.lower(): k for k in tgt}

    rows, matched = [], set()
    for cat, val in sorted(actual.items(), key=lambda kv: -kv[1]):
        t_key = lower_tgt.get(cat.lower())
        t_pct = tgt[t_key] if t_key else 0.0
        if t_key:
            matched.add(t_key)
        a_pct = val / total * 100
        rows.append((cat, a_pct, t_pct, a_pct - t_pct, val))
    for cat, t_pct in tgt.items():
        if cat not in matched:
            rows.append((cat, 0.0, t_pct, -t_pct, 0.0))

    findings = []
    t_sum = sum(tgt.values())
    if abs(t_sum - 100) > 0.01:
        findings.append(Finding(
            "allocation", "warn",
            f"Target percentages sum to {t_sum:g}, not 100 — drift below is against your numbers as given.",
            {"target_sum": t_sum},
        ))
    for cat, a, t, drift, val in rows:
        flagged = abs(drift) >= drift_threshold_pct
        findings.append(Finding(
            "allocation", "warn" if flagged else "info",
            f"{cat}: {a:.1f}% actual vs {t:.1f}% target ({drift:+.1f} pts)"
            + (" — outside threshold" if flagged else ""),
            {"category": cat, "actual_pct": round(a, 2), "target_pct": t,
             "drift_pts": round(drift, 2), "value": round(val, 2), "flagged": flagged,
             "in_target": cat.lower() in lower_tgt or t > 0},
        ))
    return findings


# ── 3. overlap (category + AMC only) ─────────────────────────────────────

def check_overlap(holdings) -> list[Finding]:
    out = []
    held = list(holdings)

    # Same scheme in more than one plan/option, or the same code in two folios.
    by_identity: dict[tuple[str, str], list[MFHolding]] = defaultdict(list)
    for h in held:
        by_identity[(h.scheme.amc_name, normalize_scheme_name(h.scheme.scheme_name))].append(h)
    for (amc, _), group in by_identity.items():
        codes = {h.scheme_id for h in group}
        if len(codes) > 1:
            names = ", ".join(sorted(f"{h.scheme.scheme_name} [{h.scheme.plan or '?'}]" for h in group))
            out.append(Finding(
                "overlap", "warn",
                f"Same underlying scheme held under {len(codes)} plan/option codes at {amc}: {names}.",
                {"amc": amc, "scheme_codes": sorted(codes), "level": "same_scheme"},
            ))

    # Several distinct schemes of one AMC in one category.
    by_cat_amc: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for h in held:
        by_cat_amc[(category_label(h.scheme), h.scheme.amc_name)][
            normalize_scheme_name(h.scheme.scheme_name)] = h.scheme.scheme_name
    for (cat, amc), schemes in by_cat_amc.items():
        if len(schemes) >= 2:
            out.append(Finding(
                "overlap", "warn",
                f"{len(schemes)} different {amc} schemes in category '{cat}': "
                + ", ".join(sorted(schemes.values())) + ". Likely overlapping portfolios.",
                {"amc": amc, "category": cat, "schemes": sorted(schemes.values()), "level": "category_amc"},
            ))

    out.append(Finding("overlap", "info", OVERLAP_OPEN_QUESTION,
                       {"level": "stock_level", "open_question": True}))
    return out


# ── 4. SIP continuity ────────────────────────────────────────────────────

def _cadence_days(gaps: list[int]) -> tuple[str, int] | None:
    m = median(gaps)
    for name, lo, hi, nominal in (
        ("weekly", 5, 9, 7), ("fortnightly", 12, 17, 14),
        ("monthly", 25, 35, 30), ("quarterly", 80, 100, 91),
    ):
        if lo <= m <= hi:
            return name, nominal
    return None


def check_sip_continuity(holdings, today: date) -> list[Finding]:
    out = []
    for h in holdings:
        dates = sorted(set(
            MFTransaction.objects.filter(holding=h, txn_type=MFTransaction.SIP)
            .values_list("txn_date", flat=True)
        ))
        if len(dates) < SIP_MIN_HISTORY:
            continue
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        cadence = _cadence_days(gaps)
        if cadence is None:
            continue
        name, nominal = cadence
        overdue_days = (today - dates[-1]).days - nominal
        if overdue_days > SIP_GRACE_DAYS:
            out.append(Finding(
                "sip_continuity", "warn",
                f"{h.scheme.scheme_name} (folio {h.folio_number or '-'}): {name} SIP last seen "
                f"{dates[-1]}, {overdue_days} days overdue — possibly lapsed (or stopped on purpose).",
                {"holding_id": h.id, "folio": h.folio_number, "cadence": name,
                 "last_sip": dates[-1].isoformat(), "days_overdue": overdue_days,
                 "sip_count": len(dates)},
            ))
    return out


# ── orchestration ────────────────────────────────────────────────────────

def run_advisor(
    tenant,
    portfolio=None,
    target: dict[str, float] | None = None,
    assumed_drag_pct: float = DEFAULT_ASSUMED_DRAG_PCT,
    drift_threshold_pct: float = DEFAULT_DRIFT_THRESHOLD_PCT,
    today: date | None = None,
    emit_events: bool = True,
) -> AdvisorReport:
    """Runs all checks for a tenant (optionally one portfolio) and, unless
    ``emit_events=False``, logs each finding via ``event_writer.emit()``."""
    today = today or date.today()
    qs = MFHolding.objects.filter(tenant=tenant).select_related("scheme")
    if portfolio is not None:
        qs = qs.filter(portfolio=portfolio)
    holdings = list(qs)

    report = AdvisorReport()
    if not holdings:
        report.notes.append("No MF holdings found for this tenant/portfolio — nothing to advise on.")
        return report

    valued, unvalued = _valued(holdings)
    report.unvalued = [f"{h.scheme.scheme_name} (folio {h.folio_number or '-'})" for h in unvalued]
    report.total_value = round(sum(v for _, _, v in valued), 2)

    report.findings += check_direct_vs_regular(valued, assumed_drag_pct)
    if target:
        report.findings += check_allocation(valued, target, drift_threshold_pct)
    else:
        report.notes.append("No target allocation given — allocation check skipped (pass --target-file).")
    report.findings += check_overlap(holdings)
    report.findings += check_sip_continuity(holdings, today)

    if emit_events:
        emit_findings(report, tenant)
    return report


def emit_findings(report: AdvisorReport, tenant) -> int:
    """Logs each finding via event_writer.emit(). Never raises; returns the
    number of events actually written."""
    written = 0
    try:
        from apps.events.models import Event
        from apps.events.services.event_writer import emit
    except Exception as exc:  # noqa: BLE001
        log.warning("mf_advisor.emit_unavailable", error=str(exc))
        return 0
    for f in report.findings:
        try:
            ev = emit(
                tenant=tenant,
                type=Event.Type.WORKFLOW_STEP_COMPLETED,
                text=f.text,
                severity="warn" if f.severity == "warn" else "info",
                actor_kind="workflow",
                step_name=f"mf_advisor.{f.kind}",
                payload={"advisor": "mf", "kind": f.kind, **f.payload},
            )
            written += 1 if ev is not None else 0
        except Exception as exc:  # noqa: BLE001
            log.warning("mf_advisor.emit_failed", kind=f.kind, error=str(exc))
    return written
