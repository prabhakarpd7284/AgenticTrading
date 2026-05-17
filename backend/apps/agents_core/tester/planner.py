"""Planner agent — turns bugs + feature requests into ordered engineering tasks.

Reads open_bugs and feature_requests. Asks Claude for the next 5-8 tasks,
each with: title · priority · scope · files-to-touch · acceptance criteria
· source (which bug or feature request it implements).

Writes Task records into the palace. Existing tasks are deduped by slug.
"""
from __future__ import annotations

import json

from . import state
from .agent_base import (
    palace_snapshot, briefing_text, parse_json_response, call_claude, claude_or_skip,
)

SYSTEM = """\
You are AlphaDesk's engineering planner. You translate open bugs and
trader feature requests into a prioritised task backlog the executor
can implement one at a time.

Each task must include:
  - title       (action-oriented, fits on one line)
  - priority    "high" | "medium" | "low"
  - source      "bug" | "feature_request" | "ad_hoc"
  - source_id   (the bug's id or feature_request's id; empty for ad_hoc)
  - scope       (1-2 sentences describing the change)
  - files       (relative paths the executor will likely touch)
  - acceptance  (one-sentence Done criteria — the tester should be able to
                 verify this automatically)

Priority rules:
  - "blocker" or "high" severity bugs → priority "high"
  - feature requests in "risk" or "report" categories that unlock decisions →
    priority "high" if no equivalent task exists; "medium" otherwise
  - everything else → "medium" / "low"

Return ONLY this JSON object — no markdown fences, no prose:

{
  "summary": "<= 200 chars on what you produced this cycle",
  "tasks": [{...}]
}

Don't propose duplicates of tasks already in the palace (compare on title).
Cap at 8 tasks per cycle.
"""


def run() -> tuple[state.MindPalace, int]:
    palace = state.load()
    if not claude_or_skip("planner", palace):
        return palace, 0

    started = state._now()
    snapshot = palace_snapshot(palace, include={
        "open_bugs", "feature_requests", "tasks", "recent_runs", "recent_agent_runs",
    })
    prompt = (
        f"{SYSTEM}\n\n---\n\nPROJECT BRIEFING:\n{briefing_text()[:4000]}\n\n"
        f"---\n\nMIND PALACE SNAPSHOT:\n{json.dumps(snapshot, indent=2)}\n\n"
        "Reply with ONLY the JSON object above."
    )
    raw = call_claude(prompt)
    parsed = parse_json_response(raw)
    if not parsed:
        state.record_agent_run(palace, agent_kind="planner", started_at=started,
                                ended_at=state._now(), ok=False,
                                summary="Claude returned non-JSON or empty response.")
        state.save(palace)
        return palace, 0

    added_ids: list[str] = []
    for t in parsed.get("tasks", [])[:8]:
        title = (t.get("title") or "").strip()
        if not title:
            continue
        task = state.upsert_task(
            palace, title=title,
            priority=str(t.get("priority", "medium")),
            scope=str(t.get("scope", ""))[:600],
            files=[str(f) for f in (t.get("files") or [])][:10],
            acceptance=str(t.get("acceptance", ""))[:240],
            source=str(t.get("source", "ad_hoc")),
            source_id=str(t.get("source_id", "")),
        )
        added_ids.append(task.id)

    summary = str(parsed.get("summary") or f"Added/updated {len(added_ids)} tasks")[:240]
    state.record_agent_run(palace, agent_kind="planner", started_at=started,
                            ended_at=state._now(), ok=True,
                            summary=summary, produced_ids=added_ids)
    palace.notes.append(f"[planner @ {started}] {summary}")
    palace.notes = palace.notes[-10:]
    state.save(palace)
    return palace, len(added_ids)
