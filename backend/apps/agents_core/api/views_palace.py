"""Two endpoints on top of the AI Tester mind palace.

  GET /api/v1/agents/palace/tasks/  →  JSON of every task + counts
  GET /board/                       →  self-contained HTML Kanban board

Both deliberately AllowAny so the board can be left open in a tab and
auto-refresh without needing a JWT.
"""
from __future__ import annotations

import json
from collections import Counter

from django.http import HttpResponse, JsonResponse
from django.views.decorators.cache import never_cache
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.agents_core.tester import state


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
    --border: #30363d;
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
  }
  html { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
      Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg);
    -webkit-font-smoothing: antialiased;
  }
  header {
    display: flex; align-items: center; gap: 16px; padding: 14px 20px;
    border-bottom: 1px solid var(--border); background: var(--surface);
    position: sticky; top: 0; z-index: 10;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
  header .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px; background: var(--surface-2);
    border: 1px solid var(--border); font-size: 12px; color: var(--fg-muted);
  }
  header .pill strong { color: var(--fg); font-weight: 600; }
  header .grow { flex: 1; }
  header button {
    background: var(--surface-2); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px;
  }
  header button:hover { background: #232a33; }
  header select, header input {
    background: var(--surface-2); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px;
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
    display: flex; flex-direction: column; max-height: calc(100vh - 130px);
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
  .col-body {
    padding: 10px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 10px;
  }
  .card {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; cursor: pointer; transition: border-color .12s, transform .12s;
  }
  .card:hover { border-color: var(--accent); transform: translateY(-1px); }
  .card .top { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .card .pri {
    font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .pri-high   { background: rgba(248,81,73,.15);  color: var(--pri-high); }
  .pri-medium { background: rgba(210,153,34,.15); color: var(--pri-medium); }
  .pri-low    { background: rgba(63,185,80,.15);  color: var(--pri-low); }
  .card .title { font-size: 13.5px; line-height: 1.35; color: var(--fg); }
  .card .meta { margin-top: 8px; font-size: 11px; color: var(--fg-subtle); display: flex; flex-wrap: wrap; gap: 6px; }
  .card .meta .tag { background: rgba(88,166,255,.1); color: var(--accent); padding: 1px 6px; border-radius: 4px; }
  .empty { padding: 16px; color: var(--fg-subtle); font-size: 12px; text-align: center; }
  .loader { padding: 30px; color: var(--fg-subtle); text-align: center; font-size: 13px; }
  .error { padding: 20px; color: var(--pri-high); font-size: 13px; }

  /* Drawer */
  .drawer-bg {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 20;
    display: none; opacity: 0; transition: opacity .15s;
  }
  .drawer-bg.open { display: block; opacity: 1; }
  .drawer {
    position: fixed; right: 0; top: 0; bottom: 0; width: min(560px, 100%);
    background: var(--bg); border-left: 1px solid var(--border);
    z-index: 21; transform: translateX(100%); transition: transform .18s ease-out;
    display: flex; flex-direction: column;
  }
  .drawer.open { transform: translateX(0); }
  .drawer header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; position: static; }
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
</style>
</head>
<body>
<header>
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
  <select id="filter-src" title="Source filter">
    <option value="">all sources</option>
    <option value="feature_request">requests</option>
    <option value="bug">bugs</option>
    <option value="ad_hoc">ad-hoc</option>
  </select>
  <input id="search" placeholder="search…" size="14" />
  <select id="refresh-int">
    <option value="0">manual</option>
    <option value="15000">15s</option>
    <option value="30000" selected>30s</option>
    <option value="60000">60s</option>
  </select>
  <button id="refresh">↻ Refresh</button>
</header>
<main>
  <div id="board" class="board"></div>
</main>

<div class="drawer-bg" id="drawer-bg"></div>
<aside class="drawer" id="drawer">
  <header style="display:flex; align-items:center; justify-content:space-between;">
    <h2 id="d-title">—</h2>
    <button class="close" id="d-close">Esc</button>
  </header>
  <div class="body" id="d-body"></div>
</aside>

<div class="footer">
  Reads <code>docs/AI_TESTER_MIND_PALACE.json</code> via
  <code>GET /api/v1/agents/palace/tasks/</code> (no auth). Updated by the
  AI Team CLI (<code>python manage.py run_ai_team</code>).
</div>

<script>
const COLUMNS = [
  { key: "open",      label: "Open",     color: "var(--col-open)" },
  { key: "proposed",  label: "Proposed", color: "var(--col-proposed)" },
  { key: "in_review", label: "In review",color: "var(--col-review)" },
  { key: "done",      label: "Done",     color: "var(--col-done)" },
];

let lastTasks = [];
let lastFetched = Date.now();

function priWeight(p) { return p === "high" ? 0 : p === "medium" ? 1 : 2; }

function relTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso); if (isNaN(+d)) return "—";
  const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s/60)}m ago`;
  if (s < 86400) return `${Math.round(s/3600)}h ago`;
  return `${Math.round(s/86400)}d ago`;
}

function renderBoard(tasks) {
  const board = document.getElementById("board");
  const filterPri = document.getElementById("filter-pri").value;
  const filterSrc = document.getElementById("filter-src").value;
  const q = document.getElementById("search").value.trim().toLowerCase();

  const filtered = tasks.filter(t =>
    (!filterPri || t.priority === filterPri) &&
    (!filterSrc || (t.source || "") === filterSrc) &&
    (!q || (t.title || "").toLowerCase().includes(q) ||
            (t.scope || "").toLowerCase().includes(q) ||
            (t.id || "").toLowerCase().includes(q))
  );

  board.innerHTML = "";
  for (const col of COLUMNS) {
    const colTasks = filtered.filter(t => t.status === col.key)
      .sort((a, b) => priWeight(a.priority) - priWeight(b.priority));
    const el = document.createElement("section");
    el.className = "col";
    el.innerHTML = `
      <div class="col-head">
        <span><span class="swatch" style="background:${col.color}"></span>${col.label}</span>
        <span class="count">${colTasks.length}</span>
      </div>
      <div class="col-body" id="col-${col.key}"></div>
    `;
    board.appendChild(el);
    const body = el.querySelector(".col-body");
    if (colTasks.length === 0) {
      body.innerHTML = `<div class="empty">No tasks</div>`;
      continue;
    }
    for (const t of colTasks) body.appendChild(card(t));
  }
}

function card(t) {
  const el = document.createElement("article");
  el.className = "card";
  el.tabIndex = 0;
  const pri = (t.priority || "medium").toLowerCase();
  const src = t.source ? `<span class="tag">${t.source}</span>` : "";
  const files = (t.files || []).length ? `<span class="tag">${(t.files||[]).length} file${t.files.length===1?"":"s"}</span>` : "";
  el.innerHTML = `
    <div class="top">
      <span class="pri pri-${pri}">${pri}</span>
    </div>
    <div class="title">${escapeHtml(t.title || t.id)}</div>
    <div class="meta">${src}${files}<span>${escapeHtml((t.id||"").slice(0,40))}</span></div>
  `;
  el.addEventListener("click", () => openDrawer(t));
  el.addEventListener("keypress", (e) => { if (e.key === "Enter") openDrawer(t); });
  return el;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[c]));
}

function openDrawer(t) {
  document.getElementById("d-title").textContent = t.title || t.id;
  const files = (t.files || []).map(f => `<span class="file">${escapeHtml(f)}</span>`).join("") || `<span class="empty">No files specified</span>`;
  document.getElementById("d-body").innerHTML = `
    <div class="sect"><h3>Status · ${t.status}</h3>
      <p><code>${escapeHtml(t.id)}</code> · priority <strong>${t.priority}</strong> · source <strong>${t.source || "—"}</strong> · created ${relTime(t.created_at)}</p>
    </div>
    <div class="sect"><h3>Scope</h3>
      <p>${escapeHtml(t.scope) || "<em>No scope captured.</em>"}</p>
    </div>
    <div class="sect"><h3>Acceptance criteria</h3>
      <p>${escapeHtml(t.acceptance) || "<em>None captured.</em>"}</p>
    </div>
    <div class="sect"><h3>Files to touch</h3>
      ${files}
    </div>
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
    lastFetched = Date.now();
    document.getElementById("total").textContent = lastTasks.length;
    document.getElementById("last-run").textContent = data.latest_agent_run
      ? `${data.latest_agent_run.agent_kind || data.latest_agent_run.kind || ""} · ${relTime(data.latest_agent_run.ended_at || data.latest_agent_run.started_at)}`
      : "—";
    renderBoard(lastTasks);
    document.getElementById("ago").textContent = "just now";
  } catch (e) {
    document.getElementById("board").innerHTML = `<div class="error">Could not load: ${escapeHtml(e.message)}.<br>Is the v2 backend up on :8000?</div>`;
  }
}

let timer = null;
function setInterval2(ms) {
  if (timer) { clearInterval(timer); timer = null; }
  if (ms > 0) timer = setInterval(fetchTasks, ms);
}

document.getElementById("refresh").onclick = fetchTasks;
document.getElementById("d-close").onclick = closeDrawer;
document.getElementById("drawer-bg").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
for (const id of ["filter-pri", "filter-src", "search"]) {
  document.getElementById(id).addEventListener("input", () => renderBoard(lastTasks));
}
document.getElementById("refresh-int").addEventListener("change", (e) => setInterval2(Number(e.target.value)));

// Tick the "refreshed Xs ago" label every 5s
setInterval(() => {
  const sec = Math.round((Date.now() - lastFetched) / 1000);
  const el = document.getElementById("ago");
  if (el) el.textContent = sec < 60 ? `${sec}s ago` : `${Math.round(sec/60)}m ago`;
}, 5000);

document.getElementById("board").innerHTML = `<div class="loader">Loading tasks…</div>`;
fetchTasks();
setInterval2(30000);
</script>
</body>
</html>
"""
