"""Executor agent — proposes a code change for the next-priority open task.

CRITICAL: this agent NEVER writes code to disk. It produces a structured
Proposal that lives in the mind palace; a human (or Claude Code in another
conversation) reviews + applies. This keeps the autonomous loop reviewable
and avoids overwrites without intent.

The executor:
  1. Picks the highest-priority OPEN task (with no existing in-flight proposal).
  2. Reads up to N relevant files listed on the task (truncated to keep the
     prompt tight).
  3. Asks Claude for {summary, files_changed[{path, change_kind, approach,
     code_outline}], risks, tests_needed}.
  4. Writes the Proposal + flips task.status -> "proposed".
"""
from __future__ import annotations

import json
from pathlib import Path

from . import state
from .agent_base import (
    palace_snapshot, briefing_text, parse_json_response, call_claude, claude_or_skip,
)

MAX_FILE_CHARS = 4_000           # truncate each file to keep the prompt manageable
MAX_FILES_PER_TASK = 5

SYSTEM = """\
You are AlphaDesk's executor. You produce a STRUCTURED CODE CHANGE PROPOSAL
for one engineering task. You do NOT write final code or diffs — you
outline what the change should be, where it should live, and why.

The proposal must include:
  - summary       (one sentence — what this change does for the operator)
  - files_changed (list of {path, change_kind: "modify"|"create",
                              approach: 1-2 sentences,
                              code_outline: pseudo-code or signature sketch})
  - risks         (1-3 bullet points — what could go wrong)
  - tests_needed  (1-3 bullet points — what the tester suite should verify)

Anchor every file path in the actual repo layout. Don't invent files. If
you need a new file, say so in change_kind: "create".

Return ONLY this JSON object — no markdown fences, no prose:

{
  "summary": "...",
  "files_changed": [...],
  "risks": [...],
  "tests_needed": [...]
}
"""


def _pick_next_task(palace: state.MindPalace) -> state.Task | None:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    candidates = [t for t in palace.tasks if t.status == "open"]
    if not candidates:
        return None
    return min(candidates, key=lambda t: (priority_order.get(t.priority, 5), t.created_at))


def _read_task_files(task: state.Task) -> str:
    if not task.files:
        return "(task didn't pin any files — executor must propose new files if needed)"
    blocks: list[str] = []
    for rel in task.files[:MAX_FILES_PER_TASK]:
        p = state._REPO_ROOT / rel
        try:
            content = p.read_text()
        except OSError:
            blocks.append(f"## {rel}\n(file not found — change_kind must be 'create' if you target this path)\n")
            continue
        snippet = content if len(content) <= MAX_FILE_CHARS else (
            content[:MAX_FILE_CHARS] + f"\n... ({len(content) - MAX_FILE_CHARS} chars truncated)"
        )
        blocks.append(f"## {rel}\n```\n{snippet}\n```\n")
    return "\n".join(blocks)


def run() -> tuple[state.MindPalace, str | None]:
    """Execute one cycle. Returns (palace, proposal_id or None)."""
    palace = state.load()
    if not claude_or_skip("executor", palace):
        return palace, None

    started = state._now()
    task = _pick_next_task(palace)
    if not task:
        state.record_agent_run(palace, agent_kind="executor", started_at=started,
                                ended_at=state._now(), ok=True,
                                summary="No open tasks to execute.")
        state.save(palace)
        return palace, None

    snapshot = palace_snapshot(palace, include={"open_bugs", "feature_requests", "tasks"})
    files_blob = _read_task_files(task)
    prompt = (
        f"{SYSTEM}\n\n---\n\nPROJECT BRIEFING:\n{briefing_text()[:3000]}\n\n"
        f"---\n\nTASK TO IMPLEMENT:\n{json.dumps({'id': task.id, 'title': task.title, 'scope': task.scope, 'acceptance': task.acceptance, 'files': task.files}, indent=2)}\n\n"
        f"---\n\nRELEVANT FILES:\n{files_blob}\n\n"
        f"---\n\nMIND PALACE (context):\n{json.dumps(snapshot, indent=2)[:3000]}\n\n"
        "Reply with ONLY the proposal JSON."
    )
    raw = call_claude(prompt)
    parsed = parse_json_response(raw)
    if not parsed:
        state.record_agent_run(palace, agent_kind="executor", started_at=started,
                                ended_at=state._now(), ok=False,
                                summary=f"Claude returned non-JSON for task {task.id}.")
        state.save(palace)
        return palace, None

    files_changed = parsed.get("files_changed") or []
    cleaned_files = []
    for fc in files_changed[:8]:
        if not isinstance(fc, dict):
            continue
        cleaned_files.append({
            "path": str(fc.get("path", ""))[:240],
            "change_kind": str(fc.get("change_kind", "modify")),
            "approach": str(fc.get("approach", ""))[:400],
            "code_outline": str(fc.get("code_outline", ""))[:1600],
        })

    proposal = state.append_proposal(
        palace, task_id=task.id,
        summary=str(parsed.get("summary", ""))[:240],
        files_changed=cleaned_files,
        risks=[str(r)[:200] for r in (parsed.get("risks") or [])][:6],
        tests_needed=[str(t)[:200] for t in (parsed.get("tests_needed") or [])][:6],
    )
    state.record_agent_run(palace, agent_kind="executor", started_at=started,
                            ended_at=state._now(), ok=True,
                            summary=f"Proposed change for task '{task.title[:120]}'",
                            produced_ids=[proposal.id])
    palace.notes.append(f"[executor @ {started}] proposed {proposal.id} for {task.id}")
    palace.notes = palace.notes[-10:]
    state.save(palace)
    return palace, proposal.id
