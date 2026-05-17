"""Mind palace — persistent JSON state for the AI Tester.

Stored at the project root as `docs/AI_TESTER_MIND_PALACE.json` so it's
version-controllable (the agent's findings + test history travel with the
repo). Keep it human-readable on disk — humans grep it to understand
"what does the tester know about the state of things?".

Token-efficiency contract:
  - On every run, load the existing mind palace FIRST so we don't re-derive
    known facts via the LLM.
  - When new bugs are discovered, dedupe by `id` (a slug of the title)
    rather than appending duplicates.
  - Only keep the last 20 test runs; older ones roll off.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Bump when the schema / scope of the mind palace meaningfully changes.
# v2: added feature_requests / tasks / proposals / agent_runs for the
#     multi-agent team (trader_user · planner · executor · tester).
# v3: briefing gained the Cockpit Catalog + echo-loop dedup rules; the
#     palace_snapshot now emits a `shipped_titles` digest so the
#     trader_user + planner can dedupe against every shipped task.
CONTEXT_VERSION = 3

# state.py → tester/ → agents_core/ → apps/ → backend/ → AgenticTrading/
_REPO_ROOT = Path(__file__).resolve().parents[4]
MIND_PALACE_PATH = _REPO_ROOT / "docs" / "AI_TESTER_MIND_PALACE.json"
BRIEFING_PATH = _REPO_ROOT / "docs" / "AI_TESTER_CLAUDE.md"
MAX_RUNS_KEPT = 20


# ─────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    id: str                          # slug of title — stable across runs
    title: str
    severity: str                    # "info" | "warning" | "high" | "blocker"
    suite: str                       # which test suite raised it
    evidence: str                    # short text explaining what failed
    suggested_fix: str | None = None # populated by --llm pass
    first_seen: str = ""
    last_seen: str = ""
    occurrences: int = 1
    status: str = "open"             # "open" | "fixed" | "wontfix"


@dataclass
class TestRun:
    id: str
    started_at: str
    ended_at: str
    total: int
    passed: int
    failed: int
    finding_ids: list[str] = field(default_factory=list)


@dataclass
class FeatureRequest:
    """A "what would a trader want next" item, produced by the trader_user agent."""
    id: str
    title: str
    rationale: str
    category: str = "general"        # "report" | "chart" | "data" | "workflow" | "general"
    status: str = "open"             # "open" | "planned" | "done" | "rejected"
    created_at: str = ""
    requested_by: str = "trader_user"


@dataclass
class Task:
    """An actionable engineering task — bug fix or feature implementation."""
    id: str
    title: str
    priority: str = "medium"         # "high" | "medium" | "low"
    scope: str = ""
    files: list[str] = field(default_factory=list)
    acceptance: str = ""
    source: str = ""                 # "bug" | "feature_request" | "ad_hoc"
    source_id: str = ""              # id of the source bug / request
    status: str = "open"             # "open" | "proposed" | "in_review" | "done" | "dropped"
    created_at: str = ""


@dataclass
class Proposal:
    """A code-change proposal produced by the executor agent. NEVER auto-applied."""
    id: str
    task_id: str
    summary: str
    files_changed: list[dict] = field(default_factory=list)   # [{path, change_kind, approach, code_outline}]
    risks: list[str] = field(default_factory=list)
    tests_needed: list[str] = field(default_factory=list)
    status: str = "proposed"         # "proposed" | "applied" | "rejected"
    created_at: str = ""


@dataclass
class AgentRunLog:
    """Meta log — every time any team agent ran."""
    id: str
    agent_kind: str                  # "trader_user" | "planner" | "executor" | "tester"
    started_at: str
    ended_at: str
    ok: bool
    summary: str = ""
    produced_ids: list[str] = field(default_factory=list)


@dataclass
class MindPalace:
    context_version: int = CONTEXT_VERSION
    fingerprint: str = ""
    last_run_at: str = ""
    open_bugs: list[Finding] = field(default_factory=list)
    fixed_bugs: list[Finding] = field(default_factory=list)
    runs: list[TestRun] = field(default_factory=list)
    # Free-form notes the LLM can append between runs without us schema-policing.
    notes: list[str] = field(default_factory=list)
    # Multi-agent team state (v2):
    feature_requests: list[FeatureRequest] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    agent_runs: list[AgentRunLog] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────
def load() -> MindPalace:
    """Read mind palace from disk. Soft-reset (preserve open_bugs, drop runs)
    when CONTEXT_VERSION changed."""
    if not MIND_PALACE_PATH.exists():
        return MindPalace()
    try:
        raw = json.loads(MIND_PALACE_PATH.read_text())
    except Exception:  # noqa: BLE001 — corrupted file should not break the runner
        return MindPalace()

    palace = MindPalace(
        context_version=int(raw.get("context_version", 0)),
        fingerprint=str(raw.get("fingerprint", "")),
        last_run_at=str(raw.get("last_run_at", "")),
        open_bugs=[Finding(**b) for b in raw.get("open_bugs", [])],
        fixed_bugs=[Finding(**b) for b in raw.get("fixed_bugs", [])],
        runs=[TestRun(**r) for r in raw.get("runs", [])],
        notes=list(raw.get("notes", [])),
        feature_requests=[FeatureRequest(**f) for f in raw.get("feature_requests", [])],
        tasks=[Task(**t) for t in raw.get("tasks", [])],
        proposals=[Proposal(**p) for p in raw.get("proposals", [])],
        agent_runs=[AgentRunLog(**a) for a in raw.get("agent_runs", [])],
    )
    if palace.context_version != CONTEXT_VERSION:
        # Soft reset — preserve all collections (they're still actionable),
        # only drop the fingerprint so the next pass re-validates the surface.
        palace.context_version = CONTEXT_VERSION
        palace.fingerprint = ""
    return palace


def save(palace: MindPalace) -> None:
    palace.last_run_at = _now()
    # Roll runs forward
    palace.runs = sorted(palace.runs, key=lambda r: r.started_at)[-MAX_RUNS_KEPT:]
    # Also roll older agent_runs and proposals so the file doesn't grow
    # unbounded over many cycles.
    palace.agent_runs = sorted(palace.agent_runs, key=lambda r: r.started_at)[-40:]
    palace.proposals = palace.proposals[-40:]
    MIND_PALACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIND_PALACE_PATH.write_text(json.dumps({
        "context_version": palace.context_version,
        "fingerprint": palace.fingerprint,
        "last_run_at": palace.last_run_at,
        "open_bugs": [asdict(b) for b in palace.open_bugs],
        "fixed_bugs": [asdict(b) for b in palace.fixed_bugs],
        "runs": [asdict(r) for r in palace.runs],
        "notes": palace.notes,
        "feature_requests": [asdict(f) for f in palace.feature_requests],
        "tasks": [asdict(t) for t in palace.tasks],
        "proposals": [asdict(p) for p in palace.proposals],
        "agent_runs": [asdict(a) for a in palace.agent_runs],
    }, indent=2))


def reset() -> None:
    if MIND_PALACE_PATH.exists():
        MIND_PALACE_PATH.unlink()


# ─────────────────────────────────────────────────────────────────────────
# Bug + run helpers
# ─────────────────────────────────────────────────────────────────────────
def slugify(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    return s[:80] or "untitled"


def upsert_finding(palace: MindPalace, *, title: str, severity: str, suite: str, evidence: str) -> Finding:
    """Add a new finding or bump occurrences on an existing one (matched by id)."""
    fid = slugify(title)
    now = _now()
    for existing in palace.open_bugs:
        if existing.id == fid:
            existing.last_seen = now
            existing.occurrences += 1
            existing.evidence = evidence  # always use the freshest evidence
            existing.severity = severity
            return existing
    # Brand-new bug
    f = Finding(id=fid, title=title, severity=severity, suite=suite,
                evidence=evidence, first_seen=now, last_seen=now,
                occurrences=1, status="open")
    palace.open_bugs.append(f)
    return f


def close_finding(palace: MindPalace, finding_id: str) -> bool:
    """Move a finding from open_bugs → fixed_bugs."""
    for i, b in enumerate(palace.open_bugs):
        if b.id == finding_id:
            b.status = "fixed"
            palace.fixed_bugs.append(b)
            del palace.open_bugs[i]
            return True
    return False


def record_run(palace: MindPalace, *, run_id: str, started_at: str, ended_at: str,
               total: int, passed: int, failed: int, finding_ids: list[str]) -> None:
    palace.runs.append(TestRun(
        id=run_id, started_at=started_at, ended_at=ended_at,
        total=total, passed=passed, failed=failed,
        finding_ids=list(finding_ids),
    ))


# ─────────────────────────────────────────────────────────────────────────
# Fingerprint — has the surface shifted since last run?
# ─────────────────────────────────────────────────────────────────────────
def compute_fingerprint(strategies: list[str], plugin_dirs: list[Path]) -> str:
    """sha256 of (sorted strategy names + plugin file mtimes). Cheap to
    compute; lets the tester short-circuit identical surfaces."""
    h = hashlib.sha256()
    for s in sorted(strategies):
        h.update(s.encode())
    for d in sorted(p.resolve() for p in plugin_dirs):
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.py")):
            try:
                h.update(str(int(f.stat().st_mtime)).encode())
                h.update(f.name.encode())
            except OSError:
                pass
    return h.hexdigest()[:16]


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────
# Multi-agent collection helpers (all dedupe by slug(title) so re-runs
# stay token-cheap and don't pollute the palace)
# ─────────────────────────────────────────────────────────────────────────
def upsert_feature_request(palace: MindPalace, *, title: str, rationale: str,
                            category: str = "general", requested_by: str = "trader_user") -> FeatureRequest:
    fid = slugify(title)
    for f in palace.feature_requests:
        if f.id == fid:
            f.rationale = rationale
            f.category = category
            return f
    f = FeatureRequest(id=fid, title=title, rationale=rationale, category=category,
                        status="open", created_at=_now(), requested_by=requested_by)
    palace.feature_requests.append(f)
    return f


def upsert_task(palace: MindPalace, *, title: str, priority: str = "medium",
                scope: str = "", files: list[str] | None = None,
                acceptance: str = "", source: str = "ad_hoc", source_id: str = "") -> Task:
    tid = slugify(title)
    for t in palace.tasks:
        if t.id == tid:
            t.priority = priority
            t.scope = scope or t.scope
            t.files = files or t.files
            t.acceptance = acceptance or t.acceptance
            return t
    t = Task(id=tid, title=title, priority=priority, scope=scope, files=files or [],
              acceptance=acceptance, source=source, source_id=source_id,
              status="open", created_at=_now())
    palace.tasks.append(t)
    return t


def mark_task_done(palace: MindPalace, task_id: str) -> bool:
    """Flip a task to `done` AND auto-close its parent feature_request.

    Returns True if the task was found. Used by the cleanup helpers and the
    executor's post-apply hook so the planner never re-tasks a shipped
    feature on the next cycle.
    """
    task = next((t for t in palace.tasks if t.id == task_id), None)
    if not task:
        return False
    task.status = "done"
    if task.source == "feature_request" and task.source_id:
        fr = next((f for f in palace.feature_requests if f.id == task.source_id), None)
        if fr and fr.status != "done":
            fr.status = "done"
    return True


def append_proposal(palace: MindPalace, *, task_id: str, summary: str,
                     files_changed: list[dict], risks: list[str] | None = None,
                     tests_needed: list[str] | None = None) -> Proposal:
    pid = f"prop-{task_id}-{len(palace.proposals)+1:03d}"
    p = Proposal(id=pid, task_id=task_id, summary=summary,
                  files_changed=files_changed, risks=risks or [],
                  tests_needed=tests_needed or [], status="proposed",
                  created_at=_now())
    palace.proposals.append(p)
    # Move related task into review state so the planner skips it next cycle.
    for t in palace.tasks:
        if t.id == task_id and t.status == "open":
            t.status = "proposed"
            break
    return p


def record_agent_run(palace: MindPalace, *, agent_kind: str, started_at: str,
                      ended_at: str, ok: bool, summary: str = "",
                      produced_ids: list[str] | None = None) -> AgentRunLog:
    a = AgentRunLog(
        id=f"{agent_kind}-{slugify(started_at)}",
        agent_kind=agent_kind, started_at=started_at, ended_at=ended_at,
        ok=ok, summary=summary, produced_ids=produced_ids or [],
    )
    palace.agent_runs.append(a)
    return a
