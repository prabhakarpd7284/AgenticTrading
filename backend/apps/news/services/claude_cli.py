"""Claude Code CLI invocation (``claude --print``, Max plan — no API credits).

Same discovery order as ``trading/agents/planner.py``:
CLAUDE_CLI_PATH → PATH → common install locations → login shell ``which``.
Runs from a temp directory so the repo's CLAUDE.md isn't pulled into every
scoring prompt.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile


class ClaudeCLIError(Exception):
    pass


def find_claude_binary() -> str:
    path = os.getenv("CLAUDE_CLI_PATH", "").strip()
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    path = shutil.which("claude") or ""
    if path:
        return path
    home = os.path.expanduser("~")
    candidates = ["/usr/local/bin/claude", "/opt/homebrew/bin/claude",
                  f"{home}/.local/bin/claude", f"{home}/.claude/local/claude",
                  f"{home}/.npm-global/bin/claude"]
    candidates += glob.glob(f"{home}/.nvm/versions/node/*/bin/claude")
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    for shell in (["bash", "-lc", "which claude"], ["zsh", "-lc", "which claude"]):
        try:
            r = subprocess.run(shell, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().splitlines()[-1]
        except Exception:  # noqa: BLE001
            continue
    raise ClaudeCLIError("Claude CLI not found (set CLAUDE_CLI_PATH)")


def run_claude(prompt: str, *, timeout: float = 240, model: str = "") -> str:
    """Send ``prompt`` on stdin to ``claude --print``; return stdout text."""
    binary = find_claude_binary()
    cmd = [binary, "--print"]
    if model:
        cmd += ["--model", model]
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)   # allow running from inside a Claude Code session
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"Claude CLI timed out after {timeout:.0f}s") from e
    except OSError as e:
        raise ClaudeCLIError(f"Claude CLI failed to start: {e}") from e
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        raise ClaudeCLIError(f"Claude CLI exit {r.returncode}: {(r.stderr or out)[:300]}")
    if not out:
        raise ClaudeCLIError("Claude CLI returned an empty response")
    if "Not logged in" in out[:200]:
        raise ClaudeCLIError("Claude CLI is not logged in")
    return out
