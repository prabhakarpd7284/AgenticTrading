"""Endpoints on top of the AI Tester mind palace.

  GET  /api/v1/agents/palace/tasks/  →  JSON of every task + counts
  POST /api/v1/agents/palace/run/    →  fires `run_ai_team --all-profiles`
                                        in a background subprocess
  GET  /api/v1/agents/palace/status/ →  is a run in progress + when did it start
  GET  /board/                       →  self-contained HTML Kanban board

All AllowAny so the board can be left open in a tab and auto-refresh
without needing a JWT.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.agents_core.tester import state


_RUN_KEY = "agents:team_run"
_RUN_TTL = 60 * 30   # 30 min — longer than any reasonable cycle


class PalaceTasksView(APIView):
    """GET /api/v1/agents/palace/tasks/ — read-only snapshot of tasks."""
    permission_classes = [AllowAny]
    authentication_classes: list = []

    def get(self, request):
        palace = state.load()
        tasks = [t.__dict__ for t in palace.tasks]
        counts = Counter(t["status"] for t in tasks)
        latest_run = palace.agent_runs[-1].__dict__ if palace.agent_runs else None
        return Response({
            "counts": dict(counts),
            "tasks": tasks,
            "feature_requests": [f.__dict__ for f in palace.feature_requests],
            "proposals": [p.__dict__ for p in palace.proposals],
            "open_bugs": [b.__dict__ for b in palace.open_bugs],
            "latest_agent_run": latest_run,
            "notes_tail": palace.notes[-3:] if palace.notes else [],
        })


@never_cache
def task_board(request):
    """Single self-contained Jira-style HTML page."""
    return HttpResponse(_BOARD_HTML, content_type="text/html; charset=utf-8")


def _is_alive(pid: int) -> bool:
    """Cheap PID liveness probe — signal 0 doesn't actually signal."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


@csrf_exempt
@never_cache
def team_run_view(request):
    """POST /api/v1/agents/palace/run/  body: {profiles?: "all" | "default" | ..., skip_tester?: bool}

    Spawns `python manage.py run_ai_team` in a detached subprocess so the
    HTTP call returns immediately. Stores the PID + start time in the
    Django cache; the status endpoint reads them to report progress.

    Refuses to start a second run while one is alive.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    existing = cache.get(_RUN_KEY)
    if existing and _is_alive(existing.get("pid", 0)):
        return JsonResponse({
            "started": False,
            "already_running": True,
            "started_at": existing.get("started_at"),
            "pid": existing.get("pid"),
        })

    body = {}
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        pass
    profiles = (body.get("profiles") or "all").lower()
    skip_tester = bool(body.get("skip_tester"))

    cmd = ["python", "manage.py", "run_ai_team"]
    if profiles == "all":
        cmd.append("--all-profiles")
    elif profiles:
        cmd.extend(["--profile", profiles])
    if skip_tester:
        cmd.append("--skip-tester")

    # Backend dir is two parents up from this file (api → agents_core → apps).
    # Walk up explicitly so the subprocess inherits the right CWD even if the
    # server was started from elsewhere.
    backend_dir = Path(__file__).resolve().parents[3]
    log_path = Path("/tmp") / "ai_team_run.log"

    env = os.environ.copy()
    env.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

    try:
        with open(log_path, "ab") as logf:
            logf.write(f"\n\n=== team run started at {datetime.now(timezone.utc).isoformat()} ===\n".encode())
            proc = subprocess.Popen(
                cmd, cwd=str(backend_dir), env=env,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,   # detach from the server process group
            )
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"started": False, "error": str(e)}, status=500)

    record = {
        "pid": proc.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cmd": " ".join(cmd),
        "log_path": str(log_path),
    }
    cache.set(_RUN_KEY, record, _RUN_TTL)
    return JsonResponse({"started": True, **record})


@never_cache
def team_status_view(request):
    """GET /api/v1/agents/palace/status/

    Returns:
      - running: bool
      - started_at / pid / cmd: when running
      - last_finished_at: latest agent_run end-timestamp from the palace
      - log_tail: last 60 lines of the subprocess log so the FE can show
                  step progress without exposing a streaming endpoint
    """
    rec = cache.get(_RUN_KEY)
    running = False
    if rec and _is_alive(rec.get("pid", 0)):
        running = True
    elif rec:
        cache.delete(_RUN_KEY)

    palace = state.load()
    latest = palace.agent_runs[-1] if palace.agent_runs else None

    log_tail = ""
    log_path = (rec or {}).get("log_path") or "/tmp/ai_team_run.log"
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 4096), os.SEEK_SET)
                log_tail = f.read().decode(errors="ignore").splitlines()[-60:]
                log_tail = "\n".join(log_tail)
            except Exception:  # noqa: BLE001
                log_tail = ""
    except FileNotFoundError:
        log_tail = ""

    return JsonResponse({
        "running": running,
        "started_at": (rec or {}).get("started_at"),
        "pid": (rec or {}).get("pid"),
        "cmd": (rec or {}).get("cmd"),
        "last_finished_agent": (latest.__dict__ if latest else None),
        "log_tail": log_tail,
    })


_BOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AlphaDesk — Task Board</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #1c232c;
    --surface-3: #232a33;
    --border: #30363d;
    --border-soft: #21262d;
    --fg: #e6edf3;
    --fg-muted: #8b949e;
    --fg-subtle: #6e7681;
    --accent: #58a6ff;
    --pri-high: #f85149;
    --pri-medium: #d29922;
    --pri-low: #3fb950;
    --col-open: #6e7681;
    --col-proposed: #a371f7;
    --col-review: #58a6ff;
    --col-done: #3fb950;
    --col-dropped: #484f58;

    /* Profile palette — one stable hue per persona */
    --p-default:   #8b949e;
    --p-options:   #a371f7;
    --p-futures:   #58a6ff;
    --p-equity:    #3fb950;
    --p-intraday:  #f0883e;
    --p-backtester:#39c5cf;
    --p-bug:       #f85149;
  }
  html { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
      Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg);
    -webkit-font-smoothing: antialiased;
  }

  header.topbar {
    display: flex; align-items: center; gap: 14px; padding: 12px 20px;
    border-bottom: 1px solid var(--border); background: var(--surface);
    position: sticky; top: 0; z-index: 20;
  }
  header.topbar h1 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px; background: var(--surface-2);
    border: 1px solid var(--border); font-size: 12px; color: var(--fg-muted);
  }
  .pill strong { color: var(--fg); font-weight: 600; }
  .grow { flex: 1; }
  header.topbar button, header.topbar select, header.topbar input {
    background: var(--surface-2); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px;
  }
  header.topbar button:hover { background: var(--surface-3); cursor: pointer; }

  .profile-strip {
    display: flex; align-items: center; gap: 6px; padding: 10px 20px;
    border-bottom: 1px solid var(--border-soft); background: var(--surface);
    overflow-x: auto;
  }
  .profile-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px;
    background: var(--surface-2); border: 1px solid var(--border);
    font-size: 12px; color: var(--fg-muted); cursor: pointer;
    white-space: nowrap; transition: all .12s;
  }
  .profile-chip:hover { color: var(--fg); border-color: var(--accent); }
  .profile-chip.active {
    color: var(--fg); background: var(--surface-3);
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent) inset;
  }
  .profile-chip .dot {
    width: 8px; height: 8px; border-radius: 999px; display: inline-block;
  }
  .profile-chip .count {
    color: var(--fg-subtle); font-size: 11px; padding-left: 4px;
    border-left: 1px solid var(--border); margin-left: 2px;
  }

  main { padding: 18px 20px 60px; }
  .board {
    display: grid; grid-template-columns: repeat(4, minmax(260px, 1fr));
    gap: 14px;
  }
  @media (max-width: 1100px) { .board { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 640px)  { .board { grid-template-columns: 1fr; } }
  .col {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    display: flex; flex-direction: column; max-height: calc(100vh - 200px);
  }
  .col-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: 13px;
  }
  .col-head .swatch { width: 8px; height: 8px; border-radius: 999px; display: inline-block; margin-right: 8px; }
  .col-head .count {
    color: var(--fg-subtle); font-weight: 500; font-size: 12px;
    background: var(--surface-2); padding: 2px 8px; border-radius: 999px;
  }
  .col-head .collapse-btn {
    background: transparent; border: none; color: var(--fg-subtle); font-size: 11px;
    cursor: pointer; padding: 2px 6px;
  }
  .col-head .collapse-btn:hover { color: var(--fg); }
  .col-body {
    padding: 10px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 10px;
  }
  .col.collapsed .col-body { display: none; }

  .card {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; cursor: pointer; transition: border-color .12s, transform .12s;
    border-left: 3px solid var(--p-default);
  }
  .card:hover { border-color: var(--accent); transform: translateY(-1px); }
  .card .top {
    display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap;
  }
  .card .pri {
    font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .pri-high   { background: rgba(248,81,73,.15);  color: var(--pri-high); }
  .pri-medium { background: rgba(210,153,34,.15); color: var(--pri-medium); }
  .pri-low    { background: rgba(63,185,80,.15);  color: var(--pri-low); }

  .profile-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; padding: 2px 7px; border-radius: 999px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .profile-badge .dot { width: 6px; height: 6px; border-radius: 999px; display: inline-block; }

  .card .title { font-size: 13.5px; line-height: 1.35; color: var(--fg); }
  .card .meta {
    margin-top: 8px; font-size: 11px; color: var(--fg-subtle);
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  }
  .card .tag {
    background: rgba(88,166,255,.1); color: var(--accent);
    padding: 1px 6px; border-radius: 4px; font-weight: 500;
  }
  .card.extend { background: linear-gradient(180deg, var(--surface-2) 0%, rgba(57, 197, 207, 0.06) 100%); }
  .card.extend .title-prefix {
    font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
    color: var(--p-backtester); margin-right: 6px;
    background: rgba(57, 197, 207, 0.12); padding: 1px 5px; border-radius: 3px;
    vertical-align: 1px;
  }

  .empty { padding: 16px; color: var(--fg-subtle); font-size: 12px; text-align: center; }
  .loader { padding: 30px; color: var(--fg-subtle); text-align: center; font-size: 13px; }
  .error { padding: 20px; color: var(--pri-high); font-size: 13px; }

  .drawer-bg {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 30;
    display: none; opacity: 0; transition: opacity .15s;
  }
  .drawer-bg.open { display: block; opacity: 1; }
  .drawer {
    position: fixed; right: 0; top: 0; bottom: 0; width: min(620px, 100%);
    background: var(--bg); border-left: 1px solid var(--border);
    z-index: 31; transform: translateX(100%); transition: transform .18s ease-out;
    display: flex; flex-direction: column;
  }
  .drawer.open { transform: translateX(0); }
  .drawer header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; }
  .drawer .body { padding: 18px 20px; overflow-y: auto; flex: 1; }
  .drawer h2 { margin: 0 0 4px; font-size: 16px; }
  .drawer .sect { margin-top: 18px; }
  .drawer .sect h3 {
    margin: 0 0 6px; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--fg-subtle); font-weight: 600;
  }
  .drawer .sect p { margin: 0; line-height: 1.5; font-size: 13.5px; color: var(--fg); }
  .drawer code, .drawer .file {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    background: var(--surface-2); padding: 2px 6px; border-radius: 4px;
    color: var(--fg); word-break: break-all;
  }
  .drawer .file { display: block; padding: 6px 8px; margin-bottom: 4px; }
  .drawer .close {
    background: transparent; border: 1px solid var(--border); color: var(--fg);
    border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 13px;
  }

  .footer {
    color: var(--fg-subtle); font-size: 11px; padding: 16px 20px;
    border-top: 1px solid var(--border); margin-top: 30px;
  }
  .ago { color: var(--accent); }

  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--fg-subtle); }
</style>
</head>
<body>
<header class="topbar">
  <h1>📋 AlphaDesk Task Board</h1>
  <span class="pill"><span id="total">—</span> tasks</span>
  <span class="pill">last run: <strong id="last-run">—</strong></span>
  <span class="pill">refreshed <span class="ago" id="ago">just now</span></span>
  <span class="grow"></span>
  <select id="filter-pri" title="Priority filter">
    <option value="">all priorities</option>
    <option value="high">high only</option>
    <option value="medium">medium only</option>
    <option value="low">low only</option>
  </select>
  <input id="search" placeholder="search…" size="14" />
  <label style="display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--fg-muted)">
    <input type="checkbox" id="show-done" /> show done
  </label>
  <select id="refresh-int">
    <option value="0">manual</option>
    <option value="15000">15s</option>
    <option value="30000" selected>30s</option>
    <option value="60000">60s</option>
  </select>
  <button id="refresh">&#8635; Refresh</button>
  <select id="run-profile" title="Profile for the team run">
    <option value="all" selected>all personas</option>
    <option value="default">default</option>
    <option value="options">options</option>
    <option value="futures">futures</option>
    <option value="equity">equity</option>
    <option value="intraday">intraday</option>
    <option value="backtester">backtester</option>
  </select>
  <button id="run-team" style="background:var(--col-done);color:#0d1117;font-weight:600;">▶ Run AI Team</button>
</header>

<div id="run-strip" style="display:none;padding:10px 20px;border-bottom:1px solid var(--border-soft);background:var(--surface);font-size:12px;color:var(--fg-muted);">
  <strong id="run-state" style="color:var(--fg);">—</strong>
  <span id="run-cmd" style="margin-left:8px;font-family:ui-monospace,Menlo,monospace;color:var(--fg-subtle);"></span>
  <pre id="run-log" style="margin:8px 0 0;padding:8px;background:var(--surface-2);border:1px solid var(--border);border-radius:6px;color:var(--fg-muted);max-height:140px;overflow:auto;font-size:11px;white-space:pre-wrap;"></pre>
</div>

<div class="profile-strip" id="profile-strip" aria-label="Filter by persona"></div>

<main>
  <div id="board" class="board"></div>
</main>

<div class="drawer-bg" id="drawer-bg"></div>
<aside class="drawer" id="drawer">
  <header style="display:flex; align-items:center; justify-content:space-between;">
    <h2 id="d-title">&mdash;</h2>
    <button class="close" id="d-close">Esc</button>
  </header>
  <div class="body" id="d-body"></div>
</aside>

<div class="footer">
  Reads <code>docs/AI_TESTER_MIND_PALACE.json</code> via
  <code>GET /api/v1/agents/palace/tasks/</code> (no auth). Updated by
  <code>python manage.py run_ai_team --all-profiles</code>.
</div>

<script>
const COLUMNS = [
  { key: "open",      label: "Open",     color: "var(--col-open)" },
  { key: "proposed",  label: "Proposed", color: "var(--col-proposed)" },
  { key: "in_review", label: "In review",color: "var(--col-review)" },
  { key: "done",      label: "Done",     color: "var(--col-done)" },
];

const PROFILES = [
  { key: "all",        label: "All",        color: "var(--accent)" },
  { key: "default",    label: "Generalist", color: "var(--p-default)" },
  { key: "options",    label: "Options",    color: "var(--p-options)" },
  { key: "futures",    label: "Futures",    color: "var(--p-futures)" },
  { key: "equity",     label: "Equity",     color: "var(--p-equity)" },
  { key: "intraday",   label: "Intraday",   color: "var(--p-intraday)" },
  { key: "backtester", label: "Backtester", color: "var(--p-backtester)" },
  { key: "bug",        label: "Bugs",       color: "var(--p-bug)" },
];

let lastTasks = [];
let lastFRs = [];
let lastFetched = Date.now();
let activeProfile = "all";
let showDone = false;
const collapsed = { done: true };

function priWeight(p) { return p === "high" ? 0 : p === "medium" ? 1 : 2; }
function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, function(c) {
    return ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" })[c];
  });
}
function relTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso); if (isNaN(+d)) return "—";
  const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s/60)}m ago`;
  if (s < 86400) return `${Math.round(s/3600)}h ago`;
  return `${Math.round(s/86400)}d ago`;
}

function taskProfile(task) {
  if (task.source === "bug") return "bug";
  if (task.source === "feature_request") {
    const fr = lastFRs.find(f => f.id === task.source_id);
    if (fr && fr.requested_by) {
      const parts = String(fr.requested_by).split(":");
      return parts.length > 1 ? parts[1] : "default";
    }
  }
  return "default";
}

function profileColor(key) {
  return `var(--p-${key === "bug" ? "bug" : key})`;
}

function renderProfileStrip() {
  const counts = {};
  for (const t of lastTasks) {
    if (!showDone && t.status === "done") continue;
    const p = taskProfile(t);
    counts[p] = (counts[p] || 0) + 1;
  }
  counts.all = Object.values(counts).reduce((a, b) => a + b, 0);
  const el = document.getElementById("profile-strip");
  el.innerHTML = PROFILES.map(p => {
    const n = counts[p.key] || 0;
    if (p.key !== "all" && n === 0) return "";
    return `<button class="profile-chip ${activeProfile === p.key ? "active" : ""}" data-key="${p.key}">
      <span class="dot" style="background:${p.color}"></span>${p.label}
      <span class="count">${n}</span>
    </button>`;
  }).join("");
  el.querySelectorAll(".profile-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      activeProfile = btn.dataset.key;
      renderProfileStrip();
      renderBoard();
    });
  });
}

function renderBoard() {
  const board = document.getElementById("board");
  const filterPri = document.getElementById("filter-pri").value;
  const q = document.getElementById("search").value.trim().toLowerCase();

  const filtered = lastTasks.filter(t => {
    if (!showDone && t.status === "done") return false;
    if (filterPri && t.priority !== filterPri) return false;
    if (activeProfile !== "all" && taskProfile(t) !== activeProfile) return false;
    if (q) {
      const hay = (t.title + " " + (t.scope || "") + " " + (t.id || "")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const visibleCols = COLUMNS.filter(c =>
    showDone || c.key !== "done" || filtered.some(t => t.status === "done"),
  );

  board.innerHTML = "";
  board.style.gridTemplateColumns = `repeat(${visibleCols.length}, minmax(260px, 1fr))`;

  for (const col of visibleCols) {
    const colTasks = filtered.filter(t => t.status === col.key)
      .sort((a, b) => priWeight(a.priority) - priWeight(b.priority));
    const isCollapsed = collapsed[col.key];
    const el = document.createElement("section");
    el.className = "col" + (isCollapsed ? " collapsed" : "");
    el.innerHTML = `
      <div class="col-head">
        <span><span class="swatch" style="background:${col.color}"></span>${col.label}</span>
        <span>
          <span class="count">${colTasks.length}</span>
          <button class="collapse-btn" data-col="${col.key}">${isCollapsed ? "▶" : "▼"}</button>
        </span>
      </div>
      <div class="col-body"></div>
    `;
    board.appendChild(el);
    const body = el.querySelector(".col-body");
    if (colTasks.length === 0) {
      body.innerHTML = `<div class="empty">No tasks</div>`;
      continue;
    }
    for (const t of colTasks) body.appendChild(card(t));
  }

  board.querySelectorAll(".collapse-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const k = btn.dataset.col;
      collapsed[k] = !collapsed[k];
      renderBoard();
    });
  });
}

function card(t) {
  const el = document.createElement("article");
  el.className = "card";
  el.tabIndex = 0;
  const isExtend = (t.title || "").startsWith("EXTEND ");
  if (isExtend) el.classList.add("extend");

  const pri = (t.priority || "medium").toLowerCase();
  const prof = taskProfile(t);
  const profMeta = PROFILES.find(p => p.key === prof) || PROFILES[0];
  el.style.borderLeftColor = profileColor(prof);

  const filesTag = (t.files || []).length
    ? `<span class="tag">${t.files.length} file${t.files.length === 1 ? "" : "s"}</span>` : "";

  const display = isExtend ? t.title.replace(/^EXTEND\s+/, "") : t.title;
  const prefix = isExtend ? `<span class="title-prefix">EXTEND</span>` : "";

  el.innerHTML = `
    <div class="top">
      <span class="pri pri-${pri}">${pri}</span>
      <span class="profile-badge" style="color:${profileColor(prof)};background:${profileColor(prof)}1a;">
        <span class="dot" style="background:${profileColor(prof)}"></span>${profMeta.label}
      </span>
    </div>
    <div class="title">${prefix}${escapeHtml(display || t.id)}</div>
    <div class="meta">
      ${t.source && t.source !== "ad_hoc" ? `<span class="tag">${escapeHtml(t.source)}</span>` : ""}
      ${filesTag}
      <span title="${escapeHtml(t.id)}">${escapeHtml((t.id || "").slice(0, 32))}${(t.id || "").length > 32 ? "…" : ""}</span>
    </div>
  `;
  el.addEventListener("click", () => openDrawer(t));
  el.addEventListener("keypress", (e) => { if (e.key === "Enter") openDrawer(t); });
  return el;
}

function openDrawer(t) {
  const prof = taskProfile(t);
  const profMeta = PROFILES.find(p => p.key === prof) || PROFILES[0];
  const fr = t.source === "feature_request" ? lastFRs.find(f => f.id === t.source_id) : null;
  document.getElementById("d-title").textContent = t.title || t.id;
  const files = (t.files || []).map(f => `<span class="file">${escapeHtml(f)}</span>`).join("")
    || `<span class="empty">No files specified</span>`;
  const requestedBy = (fr && fr.requested_by) ? fr.requested_by : (t.source === "bug" ? "tester" : "ad_hoc");
  document.getElementById("d-body").innerHTML = `
    <div class="sect"><h3>Meta</h3>
      <p>
        <code>${escapeHtml(t.id)}</code>
        · <span class="profile-badge" style="color:${profileColor(prof)};background:${profileColor(prof)}1a;">
            <span class="dot" style="background:${profileColor(prof)}"></span>${profMeta.label}
          </span>
        · priority <strong>${escapeHtml(t.priority)}</strong>
        · status <strong>${escapeHtml(t.status)}</strong>
        · created ${relTime(t.created_at)}
      </p>
      <p style="margin-top:6px;color:var(--fg-muted);font-size:13px;">
        Requested by <code>${escapeHtml(requestedBy)}</code>${fr ? ` · category: ${escapeHtml(fr.category || "?")}` : ""}
      </p>
    </div>
    ${fr ? `<div class="sect"><h3>Original request rationale</h3>
      <p style="color:var(--fg-muted);">${escapeHtml(fr.rationale || "(none)")}</p>
    </div>` : ""}
    <div class="sect"><h3>Scope</h3>
      <p>${escapeHtml(t.scope) || "<em>No scope captured.</em>"}</p>
    </div>
    <div class="sect"><h3>Acceptance criteria</h3>
      <p>${escapeHtml(t.acceptance) || "<em>None captured.</em>"}</p>
    </div>
    <div class="sect"><h3>Files to touch</h3>${files}</div>
  `;
  document.getElementById("drawer-bg").classList.add("open");
  document.getElementById("drawer").classList.add("open");
}
function closeDrawer() {
  document.getElementById("drawer-bg").classList.remove("open");
  document.getElementById("drawer").classList.remove("open");
}

async function fetchTasks() {
  try {
    const r = await fetch("/api/v1/agents/palace/tasks/", { headers: { "Accept": "application/json" } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    lastTasks = data.tasks || [];
    lastFRs = data.feature_requests || [];
    lastFetched = Date.now();
    document.getElementById("total").textContent = lastTasks.length;
    document.getElementById("last-run").textContent = data.latest_agent_run
      ? `${data.latest_agent_run.agent_kind || data.latest_agent_run.kind || ""} · ${relTime(data.latest_agent_run.ended_at || data.latest_agent_run.started_at)}`
      : "—";
    renderProfileStrip();
    renderBoard();
    document.getElementById("ago").textContent = "just now";
  } catch (e) {
    document.getElementById("board").innerHTML = `<div class="error">Could not load: ${escapeHtml(e.message)}.<br>Is the v2 backend up on :8000?</div>`;
  }
}

let timer = null;
function setRefresh(ms) {
  if (timer) { clearInterval(timer); timer = null; }
  if (ms > 0) timer = setInterval(fetchTasks, ms);
}

/* ───────────── Run AI Team button + status poller ─────────────── */
async function runTeam() {
  const profile = document.getElementById("run-profile").value;
  if (!confirm(
    `Trigger AI team cycle with profile = "${profile}"?\n\n` +
    `This takes 3-6 minutes and burns LLM credits.`
  )) return;
  const btn = document.getElementById("run-team");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    const r = await fetch("/api/v1/agents/palace/run/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ profiles: profile }),
    });
    const data = await r.json();
    if (data.error) { alert("Failed to start: " + data.error); btn.disabled = false; btn.textContent = "▶ Run AI Team"; return; }
    if (data.already_running) {
      alert("A team run is already in progress (started " + relTime(data.started_at) + ").");
    }
    startStatusPolling();
  } catch (e) {
    alert("Could not start run: " + e.message);
    btn.disabled = false; btn.textContent = "▶ Run AI Team";
  }
}

let statusTimer = null;
async function pollTeamStatus() {
  try {
    const r = await fetch("/api/v1/agents/palace/status/");
    const data = await r.json();
    const strip = document.getElementById("run-strip");
    const btn = document.getElementById("run-team");
    if (data.running) {
      strip.style.display = "";
      document.getElementById("run-state").textContent =
        "● running · started " + relTime(data.started_at) + " · pid " + data.pid;
      document.getElementById("run-cmd").textContent = data.cmd || "";
      document.getElementById("run-log").textContent = data.log_tail || "(waiting for output…)";
      btn.disabled = true; btn.textContent = "Running…";
    } else {
      if (btn.disabled) {
        // Just finished — flash a one-shot toast + refresh tasks
        strip.style.display = "";
        document.getElementById("run-state").textContent = "✓ finished";
        document.getElementById("run-log").textContent = data.log_tail || "";
        btn.disabled = false; btn.textContent = "▶ Run AI Team";
        fetchTasks();
        // Auto-hide the strip after 30s
        setTimeout(() => { strip.style.display = "none"; }, 30000);
      }
      stopStatusPolling();
    }
  } catch (e) {
    console.warn("status poll failed:", e);
  }
}
function startStatusPolling() {
  if (statusTimer) return;
  pollTeamStatus();
  statusTimer = setInterval(pollTeamStatus, 5000);
}
function stopStatusPolling() {
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}
document.getElementById("run-team").onclick = runTeam;
// Resume polling if a run is already in flight at page load
pollTeamStatus();

document.getElementById("refresh").onclick = fetchTasks;
document.getElementById("d-close").onclick = closeDrawer;
document.getElementById("drawer-bg").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
for (const id of ["filter-pri", "search"]) {
  document.getElementById(id).addEventListener("input", () => { renderProfileStrip(); renderBoard(); });
}
document.getElementById("show-done").addEventListener("change", (e) => {
  showDone = e.target.checked;
  renderProfileStrip(); renderBoard();
});
document.getElementById("refresh-int").addEventListener("change", (e) => setRefresh(Number(e.target.value)));

setInterval(() => {
  const sec = Math.round((Date.now() - lastFetched) / 1000);
  const el = document.getElementById("ago");
  if (el) el.textContent = sec < 60 ? `${sec}s ago` : `${Math.round(sec / 60)}m ago`;
}, 5000);

document.getElementById("board").innerHTML = `<div class="loader">Loading tasks…</div>`;
fetchTasks();
setRefresh(30000);
</script>
</body>
</html>
"""
