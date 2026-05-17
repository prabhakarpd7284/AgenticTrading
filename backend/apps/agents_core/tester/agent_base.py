"""Shared infrastructure for the multi-agent team.

Each agent (trader_user · planner · executor · tester) is a small wrapper
around: read mind palace → build a focused context blob → invoke Claude CLI →
parse JSON → write back to palace → log the run.

Keeping the loop deterministic + reviewable: agents NEVER apply code changes.
They produce proposals; humans (or `claude` Code) apply.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import state
from .llm import _find_claude, _invoke_claude  # reuse the existing wrappers


# Maximum payload chunks we'll show the agent — keeps the prompt tight.
TOP_N_BUGS = 10
TOP_N_REQUESTS = 8
TOP_N_TASKS = 12
TOP_N_RUNS = 5


def palace_snapshot(palace: state.MindPalace, *, include: set[str] | None = None) -> dict:
    """Token-economical view of the mind palace. Pass `include` to limit
    which collections show up.

    Defaults to a sensible "what's actionable right now" snapshot.
    """
    include = include or {"open_bugs", "feature_requests", "tasks", "recent_runs", "recent_agent_runs", "notes"}
    out: dict[str, Any] = {
        "fingerprint": palace.fingerprint,
        "last_run_at": palace.last_run_at,
    }
    if "open_bugs" in include:
        out["open_bugs"] = [
            {"id": b.id, "title": b.title, "severity": b.severity, "suite": b.suite,
             "evidence": b.evidence[:240], "occurrences": b.occurrences,
             "suggested_fix": (b.suggested_fix or "")[:240]}
            for b in palace.open_bugs[:TOP_N_BUGS]
        ]
    if "feature_requests" in include:
        out["feature_requests"] = [
            {"id": f.id, "title": f.title, "category": f.category,
             "rationale": f.rationale[:200], "status": f.status}
            for f in palace.feature_requests[-TOP_N_REQUESTS:]
        ]
    if "tasks" in include:
        out["tasks"] = [
            {"id": t.id, "title": t.title, "priority": t.priority,
             "status": t.status, "source": t.source, "scope": t.scope[:160],
             "files": t.files[:6]}
            for t in palace.tasks[-TOP_N_TASKS:]
        ]
    if "recent_runs" in include:
        out["recent_runs"] = [
            {"id": r.id, "started_at": r.started_at, "passed": r.passed,
             "failed": r.failed, "total": r.total}
            for r in palace.runs[-TOP_N_RUNS:]
        ]
    if "recent_agent_runs" in include:
        out["recent_agent_runs"] = [
            {"agent_kind": a.agent_kind, "started_at": a.started_at,
             "ok": a.ok, "summary": (a.summary or "")[:160]}
            for a in palace.agent_runs[-TOP_N_RUNS:]
        ]
    if "notes" in include:
        out["notes"] = palace.notes[-5:]
    return out


def briefing_text() -> str:
    try:
        return state.BRIEFING_PATH.read_text()
    except OSError:
        return "(briefing missing)"


def parse_json_response(raw: str) -> dict | None:
    """Tolerant JSON parse — strips markdown fences + leading prose if any."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    # Find first { and last } for safety
    m_start = cleaned.find("{")
    m_end = cleaned.rfind("}")
    if m_start < 0 or m_end < m_start:
        return None
    try:
        return json.loads(cleaned[m_start:m_end + 1])
    except json.JSONDecodeError:
        return None


def call_claude(prompt: str, *, model: str | None = None) -> str:
    claude = _find_claude()
    if not claude:
        return ""
    return _invoke_claude(claude, prompt, model=model)


def claude_or_skip(agent_kind: str, palace: state.MindPalace) -> str | None:
    """Returns the claude binary path if available; otherwise logs a skipped
    agent_run and returns None so the caller exits cleanly."""
    claude = _find_claude()
    if not claude:
        now = state._now()
        state.record_agent_run(
            palace, agent_kind=agent_kind, started_at=now, ended_at=now,
            ok=False, summary="Skipped — Claude CLI not on PATH.",
        )
        state.save(palace)
        return None
    return claude
