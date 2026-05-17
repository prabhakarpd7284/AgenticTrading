"""Trader User agent — generates feature requests from a trader's POV.

Reads the mind palace and asks Claude: "if you were the trader using
AlphaDesk right now, what reports / charts / data views would you want
next given what's already there and what's broken?"

Writes structured feature_requests back into the palace.
"""
from __future__ import annotations

import json

from . import state
from .agent_base import (
    palace_snapshot, briefing_text, parse_json_response, call_claude, claude_or_skip,
)

SYSTEM = """\
You are a quant trader using AlphaDesk for live + paper Indian-market
trading (equity intraday + monthly F&O + weekly index options). Your job:
look at what the app already does, look at the bugs that are open, and
imagine the next 3-5 reports / charts / data views you'd want so you can
see the WHOLE picture of your plan, capital deployed, leverage, and edge.

Be specific. Each request must answer:
  - WHAT screen/view/chart/report
  - WHY it matters (what decision it unlocks)

Return ONLY this JSON object — no markdown fences, no prose:

{
  "summary": "<= 200 chars on what you focused on this cycle",
  "requests": [
    {"title": "<concise, action-oriented>", "rationale": "<2-3 sentences>",
     "category": "report|chart|data|workflow|risk"}
  ]
}

Avoid duplicates of existing feature_requests in the palace. Don't suggest
things that are already in the active feature set unless they're flagged
as needing improvement in open_bugs.
"""


def run() -> tuple[state.MindPalace, int]:
    """Execute one cycle of the trader_user agent. Returns (palace, added_count)."""
    palace = state.load()
    if not claude_or_skip("trader_user", palace):
        return palace, 0

    started = state._now()
    snapshot = palace_snapshot(palace)
    prompt = (
        f"{SYSTEM}\n\n---\n\nPROJECT BRIEFING:\n{briefing_text()[:4000]}\n\n"
        f"---\n\nMIND PALACE SNAPSHOT:\n{json.dumps(snapshot, indent=2)}\n\n"
        "Reply with ONLY the JSON object above."
    )
    raw = call_claude(prompt)
    parsed = parse_json_response(raw)
    if not parsed:
        state.record_agent_run(palace, agent_kind="trader_user", started_at=started,
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
        )
        added_ids.append(f.id)

    summary = str(parsed.get("summary") or f"Added {len(added_ids)} feature requests")[:240]
    state.record_agent_run(palace, agent_kind="trader_user", started_at=started,
                            ended_at=state._now(), ok=True,
                            summary=summary, produced_ids=added_ids)
    if parsed.get("summary"):
        palace.notes.append(f"[trader_user @ {started}] {summary}")
        palace.notes = palace.notes[-10:]
    state.save(palace)
    return palace, len(added_ids)
