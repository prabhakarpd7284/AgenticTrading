"""Optional Claude CLI summary layer for the AI Tester.

When `python manage.py run_ai_tester --llm` is passed, the runner pipes
the test results + open findings + the briefing into `claude --print`
and asks for a plain-English summary plus per-bug fix proposals. The
proposals are written back into each finding's `suggested_fix`.

We use the Claude CLI exactly like the legacy planner does — no API key
required. Falls back gracefully (no LLM enrichment, no error) when the
CLI isn't on PATH.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from . import state


SYSTEM_PREAMBLE = """\
You are AlphaDesk's AI Tester analyst. You read test results from a
deterministic runner and produce ONE compact JSON object — no prose, no
markdown fences.

Schema:

{
  "summary": "<= 280 chars, what the run found at a glance",
  "fix_proposals": [
    {"finding_id": "<slug>", "fix": "<single paragraph, concrete change suggestion>"}
  ],
  "notes": ["<= 3 short notes worth remembering for next run"]
}

Be concrete. Reference file paths + function names when proposing fixes.
"""


def enrich_palace(palace: state.MindPalace, *, model: str | None = None) -> bool:
    """Mutate `palace` in place with Claude's summary + per-bug fix proposals.
    Returns True if the LLM was actually invoked."""
    claude = _find_claude()
    if not claude:
        return False

    # Read the briefing so Claude has the same context the runner used.
    try:
        briefing = state.BRIEFING_PATH.read_text()
    except OSError:
        briefing = "(briefing not found)"

    body = {
        "open_bugs": [
            {"id": b.id, "title": b.title, "suite": b.suite,
             "severity": b.severity, "evidence": b.evidence,
             "occurrences": b.occurrences, "first_seen": b.first_seen}
            for b in palace.open_bugs
        ],
        "last_run": palace.runs[-1].__dict__ if palace.runs else {},
        "fingerprint": palace.fingerprint,
    }
    prompt = (
        f"{SYSTEM_PREAMBLE}\n\n---\n\n"
        f"PROJECT BRIEFING (read-only context):\n{briefing}\n\n"
        f"---\n\nLATEST TEST RESULTS:\n{json.dumps(body, indent=2)}\n\n"
        "Respond with ONLY the JSON object described above."
    )

    raw = _invoke_claude(claude, prompt, model=model)
    if not raw:
        return False

    try:
        # Strip markdown fences if Claude added them.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").lstrip("json").strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        palace.notes.append(f"LLM enrichment returned non-JSON; kept findings as-is. Raw[:200]={raw[:200]!r}")
        return False

    # Stash the summary as the most recent note.
    summary = parsed.get("summary")
    if summary:
        palace.notes.append(f"[LLM @ {state._now()}] {summary}")
        palace.notes = palace.notes[-10:]
    # Stash any free-form notes from the LLM.
    for n in parsed.get("notes", [])[:3]:
        palace.notes.append(f"[LLM-note] {n}")
    palace.notes = palace.notes[-10:]
    # Per-finding fix proposals.
    for proposal in parsed.get("fix_proposals", []):
        fid = proposal.get("finding_id")
        fix = proposal.get("fix")
        if not fid or not fix:
            continue
        for b in palace.open_bugs:
            if b.id == fid:
                b.suggested_fix = fix
                break
    return True


# ─────────────────────────────────────────────────────────────────────────
def _find_claude() -> str | None:
    p = os.getenv("CLAUDE_CLI_PATH", "").strip() or shutil.which("claude") or ""
    if p:
        return p
    for shell_cmd in (["zsh", "-lc", "which claude"], ["bash", "-lc", "which claude"]):
        try:
            r = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:  # noqa: BLE001
            continue
    return None


def _invoke_claude(claude_path: str, prompt: str, *, model: str | None) -> str:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as t:
            t.write(prompt)
            tmp_path = t.name
        flag = f"--model {model}" if model else ""
        cmd = f'"{claude_path}" --print {flag} < "{tmp_path}"'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            env=env, timeout=120)
        if r.returncode != 0:
            return ""
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass
