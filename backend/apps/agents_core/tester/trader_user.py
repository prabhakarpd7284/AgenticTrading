"""Trader User agent — generates feature requests from a trader's POV.

Reads the mind palace and asks Claude: "if you were the trader using
AlphaDesk right now, what reports / charts / data views would you want
next given what's already there and what's broken?"

Multiple **profiles** are available — each one is a persona with a
different system prompt (generalist, options, futures, equity swing,
intraday scalper). Every request the agent writes is tagged with the
profile name on `requested_by` so the planner downstream can group them.

Writes structured feature_requests back into the palace.
"""
from __future__ import annotations

import json

from . import state
from .agent_base import (
    palace_snapshot, briefing_text, parse_json_response, call_claude, claude_or_skip,
)
from .trader_profiles import PROFILES, TraderProfile, get as get_profile


def run(profile: str = "default") -> tuple[state.MindPalace, int]:
    """Execute one cycle of the trader_user agent for a given profile.

    Returns (palace, added_count). When ``profile`` is unknown, falls
    back to ``default`` so callers (CLI / team orchestrator) can pass
    user input straight through without validation upfront.
    """
    prof: TraderProfile = get_profile(profile)
    palace = state.load()
    agent_kind = f"trader_user:{prof.id}"
    if not claude_or_skip(agent_kind, palace):
        return palace, 0

    started = state._now()
    snapshot = palace_snapshot(palace)
    prompt = (
        f"{prof.system_prompt}\n\n---\n\nPROJECT BRIEFING:\n{briefing_text()[:4000]}\n\n"
        f"---\n\nMIND PALACE SNAPSHOT:\n{json.dumps(snapshot, indent=2)}\n\n"
        "Reply with ONLY the JSON object above."
    )
    raw = call_claude(prompt)
    parsed = parse_json_response(raw)
    if not parsed:
        state.record_agent_run(palace, agent_kind=agent_kind, started_at=started,
                                ended_at=state._now(), ok=False,
                                summary="Claude returned non-JSON or empty response.")
        state.save(palace)
        return palace, 0

    added_ids: list[str] = []
    for req in parsed.get("requests", [])[:8]:
        title = (req.get("title") or "").strip()
        if not title:
            continue
        f = state.upsert_feature_request(
            palace, title=title,
            rationale=str(req.get("rationale", ""))[:600],
            category=str(req.get("category", "general")),
            requested_by=prof.requested_by,
        )
        added_ids.append(f.id)

    summary = str(parsed.get("summary") or f"[{prof.id}] Added {len(added_ids)} feature requests")[:240]
    state.record_agent_run(palace, agent_kind=agent_kind, started_at=started,
                            ended_at=state._now(), ok=True,
                            summary=summary, produced_ids=added_ids)
    if parsed.get("summary"):
        palace.notes.append(f"[{agent_kind} @ {started}] {summary}")
        palace.notes = palace.notes[-10:]
    state.save(palace)
    return palace, len(added_ids)


def run_all_profiles(profiles: list[str] | None = None) -> dict[str, int]:
    """Run every (or a chosen subset of) profiles in sequence.

    Returns {profile_id: added_count}.
    """
    ids = profiles or list(PROFILES.keys())
    out: dict[str, int] = {}
    for pid in ids:
        _, added = run(profile=pid)
        out[pid] = added
    return out
