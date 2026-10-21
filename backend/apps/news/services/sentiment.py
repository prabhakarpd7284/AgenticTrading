"""Headline sentiment via Claude CLI + deterministic per-stock rollup.

Scoring (LLM): batches of headlines → strict JSON
    [{"id", "score" −1..+1, "materiality", "confidence" 0..1}]
Unparseable / missing / timed-out → the headline stays unscored. The LLM
output can only ever feed the *blocking* filter (TRADING_RULES.md §3) — it
never sizes or forces a trade.

Rollup (deterministic): weight = recency × materiality × confidence,
    recency  = 0.5 ** (age_hours / HALF_LIFE_HOURS)
    score    = Σ w·s / Σ w
    label    = strong_* when |score| ≥ STRONG_THRESHOLD and Σ w ≥ MIN_STRONG_WEIGHT,
               positive/negative when |score| ≥ MILD_THRESHOLD, else neutral;
               "unscored" when the stock had headlines but none could be scored.
Thresholds live in ``settings.NEWS_SENTIMENT``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import structlog
from django.conf import settings
from django.utils import timezone

from apps.news.models import NewsItem
from apps.news.services import claude_cli
from apps.news.services.store import MARKET

log = structlog.get_logger(__name__)

DEFAULTS = {
    "STRONG_THRESHOLD": 0.5,
    "MILD_THRESHOLD": 0.15,
    "MIN_STRONG_WEIGHT": 0.5,
    "HALF_LIFE_HOURS": 24.0,
    "BATCH_SIZE": 30,
    "CLI_TIMEOUT_SECS": 240,
    "MAX_BATCHES": 30,
    "MAX_CONSECUTIVE_FAILURES": 2,
    "MODEL": "",           # "" = the CLI default; e.g. "sonnet" / "haiku"
    "LOOKBACK_DAYS": 3,
}

MATERIALITY_WEIGHT = {
    "results": 1.0, "guidance": 1.0, "regulatory": 0.9, "order": 0.8,
    "rating": 0.7, "management": 0.6, "other": 0.4, "macro": 0.3, "noise": 0.0,
}
VALID_MATERIALITY = set(MATERIALITY_WEIGHT)


def cfg(key: str):
    return getattr(settings, "NEWS_SENTIMENT", {}).get(key, DEFAULTS[key])


# ── Prompt + parsing ─────────────────────────────────────────────────────

PROMPT_HEADER = """You rate Indian stock-market news headlines (NSE) for short-term price impact.
For EACH headline below, judge the likely effect on the listed stock's price over the
NEXT trading day (for MARKET rows: on the Indian equity market overall).

Return ONLY a JSON array — no prose, no markdown fences — with exactly one object per
input id, in this shape:
[{"id": 1, "score": 0.0, "materiality": "other", "confidence": 0.5}]

- score: float from -1 (very negative) through 0 (neutral) to +1 (very positive).
- materiality: one of results | order | guidance | rating | regulatory | management |
  macro | other | noise
  (results = quarterly results/earnings; order = order wins/contracts/deals;
  guidance = company outlook/targets; rating = broker or credit rating/target-price
  changes; regulatory = regulator/legal/tax/government action; management =
  leadership/promoter actions; macro = economy, sector or market-wide; other = minor
  company news; noise = routine filings, listicles, no price relevance).
- confidence: float 0..1 — how sure you are of the score.
Judge only from the headline text; do not invent facts.

Headlines (id | stock | source | published | headline):
"""


def build_prompt(batch: list[tuple[int, str, str, str, str]]) -> str:
    lines = [f"{i} | {sym} | {src} | {when} | {title.replace('|', '/')}"
             for i, sym, src, when, title in batch]
    return PROMPT_HEADER + "\n".join(lines) + "\n"


def parse_scores(raw: str) -> dict[int, dict]:
    """Robustly pull ``{id: {score, materiality, confidence}}`` from CLI text.

    Accepts fenced JSON, prose around the array, a top-level object with a
    list value, and clamps/normalizes every field. Bad entries are dropped.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise ValueError(f"no JSON array in response: {text[:120]!r}")

    out: dict[int, dict] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        try:
            i = int(obj.get("id"))
            score = float(obj.get("score"))
        except (TypeError, ValueError):
            continue
        if score != score:  # NaN
            continue
        mat = str(obj.get("materiality") or "other").strip().lower()
        if mat not in VALID_MATERIALITY:
            mat = "other"
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out[i] = {
            "score": max(-1.0, min(1.0, score)),
            "materiality": mat,
            "confidence": max(0.0, min(1.0, conf if conf == conf else 0.5)),
        }
    return out


# ── Scoring run ──────────────────────────────────────────────────────────

@dataclass
class ScoreRunStatus:
    candidates: int = 0
    scored: int = 0
    batches: int = 0
    failed_batches: int = 0
    aborted: bool = False
    error: str = ""
    model: str = ""
    secs: float = 0.0

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score_items(items: list[NewsItem], *, runner=None) -> ScoreRunStatus:
    """Score ``items`` (unscored ones only) in batches through the CLI."""
    runner = runner or claude_cli.run_claude
    st = ScoreRunStatus(model=f"claude-cli:{cfg('MODEL') or 'default'}")
    todo = [it for it in items if it.scored_at is None]
    st.candidates = len(todo)
    if not todo:
        return st
    t0 = time.monotonic()
    size = max(1, int(cfg("BATCH_SIZE")))
    consecutive = 0
    for start in range(0, len(todo), size):
        if st.batches >= int(cfg("MAX_BATCHES")):
            st.error = f"stopped at MAX_BATCHES={cfg('MAX_BATCHES')}"
            break
        chunk = todo[start:start + size]
        st.batches += 1
        rows = []
        for n, it in enumerate(chunk, start=1):
            syms = ",".join(sorted(m.symbol for m in it.mentions.all())) or MARKET
            when = timezone.localtime(it.published_at).strftime("%Y-%m-%d %H:%M")
            rows.append((n, syms, it.publisher or it.source, when, it.title))
        try:
            raw = runner(build_prompt(rows), timeout=float(cfg("CLI_TIMEOUT_SECS")),
                         model=cfg("MODEL"))
            scores = parse_scores(raw)
        except Exception as e:  # noqa: BLE001 — CLI/parse failure → items stay unscored
            st.failed_batches += 1
            st.error = f"{type(e).__name__}: {e}"[:300]
            log.warning("news.sentiment.batch_failed", batch=st.batches, error=str(e))
            consecutive += 1
            if consecutive >= int(cfg("MAX_CONSECUTIVE_FAILURES")):
                st.aborted = True
                break
            continue
        consecutive = 0
        now = timezone.now()
        for n, it in enumerate(chunk, start=1):
            s = scores.get(n)
            if not s:
                continue
            it.sentiment_score = Decimal(str(round(s["score"], 3)))
            it.materiality = s["materiality"]
            it.confidence = Decimal(str(round(s["confidence"], 3)))
            it.sentiment_model = st.model[:64]
            it.scored_at = now
            it.save(update_fields=["sentiment_score", "materiality", "confidence",
                                   "sentiment_model", "scored_at"])
            st.scored += 1
    st.secs = time.monotonic() - t0
    return st


# ── Rollup ───────────────────────────────────────────────────────────────

@dataclass
class Rollup:
    score: float | None
    label: str
    weight: float
    headline_count: int
    scored_count: int


def label_for(score: float | None, weight: float, *, headline_count: int,
              scored_count: int) -> str:
    if headline_count and not scored_count:
        return "unscored"
    if score is None or weight <= 0:
        return "neutral"
    strong, mild = float(cfg("STRONG_THRESHOLD")), float(cfg("MILD_THRESHOLD"))
    can_be_strong = weight >= float(cfg("MIN_STRONG_WEIGHT"))
    if score >= strong and can_be_strong:
        return "strong_positive"
    if score <= -strong and can_be_strong:
        return "strong_negative"
    if score >= mild:
        return "positive"
    if score <= -mild:
        return "negative"
    return "neutral"


def rollup(items: list[NewsItem], *, now: datetime | None = None) -> Rollup:
    """Recency- and materiality-weighted sentiment for one stock (or MARKET)."""
    now = now or timezone.now()
    half_life = float(cfg("HALF_LIFE_HOURS"))
    relevant = [it for it in items if it.materiality != NewsItem.Materiality.NOISE
                or it.scored_at is None]
    num = den = 0.0
    scored = 0
    for it in relevant:
        if it.scored_at is None or it.sentiment_score is None:
            continue
        scored += 1
        age_h = max(0.0, (now - it.published_at).total_seconds() / 3600)
        w = (0.5 ** (age_h / half_life)) * MATERIALITY_WEIGHT.get(it.materiality or "other", 0.4) \
            * float(it.confidence if it.confidence is not None else 0.5)
        num += w * float(it.sentiment_score)
        den += w
    score = (num / den) if den > 0 else None
    label = label_for(score, den, headline_count=len(relevant), scored_count=scored)
    return Rollup(score=None if score is None else round(score, 3), label=label,
                  weight=round(den, 3), headline_count=len(relevant), scored_count=scored)
