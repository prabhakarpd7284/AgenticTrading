#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AlphaDesk — start the local dev processes in one shot.
#
#   :8000   Django ASGI         (Daphne — REST + WebSocket + broker WS)
#   :5173   Vite (React SPA)
#   —       Celery worker       (order outbox, agent runs, snapshots)
#   —       Celery beat         (periodic broker refresh, outbox poll, etc)
#
# Postgres + Redis live in docker-compose.dev.yml — bring them up separately:
#     docker compose -f docker-compose.dev.yml up -d
#
# Usage:
#     bash scripts/dev_up.sh                 # start everything
#     bash scripts/dev_up.sh --no-front      # backend + worker + beat only
#     bash scripts/dev_up.sh --no-celery     # web + ui only (no async tasks)
#     bash scripts/dev_up.sh --no-verify     # skip post-launch port-bind check
#
# GUARANTEES (idempotent restart):
#   - Every run reliably kills any prior instance — by pidfile, by process
#     name (celery), and by listening port (web :8000, vite :5173).
#   - SIGTERM → 5s grace → SIGKILL fallback for stubborn processes.
#   - Script exits non-zero if a port can't be freed or the new web process
#     fails to bind (rather than silently leaving you talking to a stale
#     instance — the failure mode that prompted this rewrite).
#   - Each process writes to logs/<name>.log; its PID to logs/<name>.pid.
#
# Stop everything cleanly with: bash scripts/dev_down.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
VENV_PY="$BACKEND_DIR/.venv/bin/python"
VENV_DAPHNE="$BACKEND_DIR/.venv/bin/daphne"
VENV_CELERY="$BACKEND_DIR/.venv/bin/celery"

# --- flags -----------------------------------------------------------------
WITH_FRONT=1
WITH_CELERY=1
WITH_VERIFY=1
for arg in "$@"; do
    case "$arg" in
        --no-front)  WITH_FRONT=0 ;;
        --no-celery) WITH_CELERY=0 ;;
        --no-verify) WITH_VERIFY=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# --- sanity ---------------------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found." >&2
    echo "       Set up the backend first:  cd backend && uv sync && uv pip install -e ." >&2
    exit 1
fi
if [[ "$WITH_FRONT" == "1" && ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "WARN:  $FRONTEND_DIR/node_modules missing — run 'cd frontend && npm install'" >&2
fi

# Quick hint if the infra containers aren't up — non-fatal.
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^alphadesk-pg$'; then
    echo "HINT:  Postgres container 'alphadesk-pg' isn't running. Start infra with:" >&2
    echo "         docker compose -f docker-compose.dev.yml up -d" >&2
fi

# ── kill helpers ──────────────────────────────────────────────────────────
# Three independent ways to kill prior runs, applied in order. Each is
# best-effort and idempotent so the script can re-run cleanly even if a
# previous shell crashed mid-launch.

# 1. Pidfile-based kill — fastest, most specific. Used as the primary
#    signal so we don't always have to scan ports / process tables.
stop_pid() {
    local pidfile="$1"
    local label="$2"
    [[ -f "$pidfile" ]] || return 0
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "  $label → kill pid=$pid (from $pidfile)"
        kill -TERM "$pid" 2>/dev/null || true
        # 3s grace then escalate.
        for _ in 1 2 3; do
            sleep 1
            kill -0 "$pid" 2>/dev/null || break
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "    pid=$pid ignored SIGTERM — sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pidfile"
}

# 2. Process-name kill — catches orphans whose pidfile was lost (common
#    after IDE force-quits, dev_up.sh interrupted mid-launch, etc).
kill_by_name() {
    local pattern="$1"
    local label="$2"
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
        return 0
    fi
    echo "  $label → kill stray processes matching '$pattern'"
    pkill -TERM -f "$pattern" 2>/dev/null || true
    sleep 2
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        pkill -KILL -f "$pattern" 2>/dev/null || true
        sleep 1
    fi
}

# 3. Port-based kill — catches a process bound to the port but invisible
#    to the above (e.g. a manual `daphne -p 8000 ...` run from another
#    shell, a Python -m http.server squatting on :5173). This is the
#    safety net that GUARANTEES the new instance can bind. SIGTERM →
#    5s grace → SIGKILL fallback. Empty PID list = nothing to do.
kill_port() {
    local port="$1"
    local label="$2"
    command -v lsof >/dev/null 2>&1 || return 0
    local pids
    pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
    [[ -z "${pids:-}" ]] && return 0
    echo "  $label :$port → kill stray pid(s): $(echo $pids | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    # Up to 5s for graceful exit.
    for _ in 1 2 3 4 5; do
        sleep 1
        pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
        [[ -z "${pids:-}" ]] && return 0
    done
    # Stubborn — escalate.
    echo "    $label :$port still held after 5s — sending SIGKILL"
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null || true
    sleep 1
}

# Block until a port is verifiably free. Fatal on timeout — if we can't
# free the port, launching a new process would silently fail (the old
# one keeps serving), which is exactly the bug this script exists to
# prevent. Better to fail loud than fall back to that.
wait_port_free() {
    local port="$1"
    local label="$2"
    command -v lsof >/dev/null 2>&1 || return 0
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        local pids
        pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
        [[ -z "${pids:-}" ]] && return 0
        sleep 0.5
    done
    local holders
    holders="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
    echo "FATAL: $label :$port is still held after kill attempt: pids=$holders" >&2
    echo "       Inspect with:  lsof -i :$port" >&2
    return 1
}

# After launching, wait until the new process is actually accepting TCP
# connections on its port. If it doesn't come up, dump the tail of its
# log so the operator sees the import error immediately.
wait_port_listen() {
    local port="$1"
    local label="$2"
    local logfile="$3"
    local timeout="${4:-20}"
    local i=0
    while ((i < timeout * 2)); do
        if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return 0
        fi
        sleep 0.5
        ((i++)) || true
    done
    echo "ERROR: $label :$port did not come up within ${timeout}s." >&2
    echo "       Last 30 lines of $logfile:" >&2
    echo "       ────────────────────────────────────────────────────" >&2
    tail -n 30 "$logfile" 2>/dev/null | sed 's/^/       │ /' >&2
    echo "       ────────────────────────────────────────────────────" >&2
    return 1
}

# ── nuke leftovers ────────────────────────────────────────────────────────
echo "[reset] clearing prior instances"
stop_pid logs/web.pid     "web"
stop_pid logs/celery.pid  "celery worker"
stop_pid logs/beat.pid    "celery beat"
stop_pid logs/vite.pid    "vite"

kill_by_name "celery -A config worker" "celery worker"
kill_by_name "celery -A config beat"   "celery beat"

# A previous run may have started Django via runserver (older versions of
# this script) or daphne (current). Either way, anything bound to :8000
# blocks us. Same for vite on :5173.
kill_port 8000 "web"
kill_port 5173 "vite"

wait_port_free 8000 "web" || exit 1
wait_port_free 5173 "vite" || exit 1

# --- Django ASGI on :8000 -------------------------------------------------
# Daphne is the ASGI server — `runserver` is WSGI and would silently
# break /ws/* WebSockets.
echo "[1/3] starting Daphne ASGI  → http://localhost:8000"
if [[ ! -x "$VENV_DAPHNE" ]]; then
    echo "ERROR: $VENV_DAPHNE not found — run 'cd backend && uv sync' to install daphne." >&2
    exit 1
fi
(
    cd "$BACKEND_DIR"
    DJANGO_SETTINGS_MODULE=config.settings.dev \
        "$VENV_DAPHNE" -b 0.0.0.0 -p 8000 config.asgi:application \
        >"$ROOT/logs/web.log" 2>&1 &
    echo $! >"$ROOT/logs/web.pid"
)

# Verify the new web process actually bound to :8000. Without this check,
# a Python import error during startup (e.g. a typo in a urls.py edit)
# leaves the script reporting success while the port is empty and the
# UI just gets ECONNREFUSED. We fail loudly here so the operator sees
# exactly what went wrong, including the tail of the log.
if [[ "$WITH_VERIFY" == "1" ]]; then
    if ! wait_port_listen 8000 "web" "$ROOT/logs/web.log" 25; then
        echo "FATAL: Daphne didn't come up. Aborting startup." >&2
        # The launching subshell already wrote logs/web.pid; clear it so
        # the next dev_up.sh won't try to kill a non-existent process.
        rm -f "$ROOT/logs/web.pid"
        exit 1
    fi
    echo "       web is live on :8000"
fi

# --- Celery worker --------------------------------------------------------
if [[ "$WITH_CELERY" == "1" ]]; then
    echo "[2/4] starting Celery worker"
    # NOTE on -Q: three tasks declare custom queues:
    #   apps.agents_core.tasks.run.execute_run     → "agents"
    #   apps.trading.tasks.outbox.process_outbox   → "orders"
    #   apps.strategies.tasks.backtest.*           → "backtests"
    # The default celery worker only consumes the "celery" queue, so without
    # naming these explicitly here, every API-triggered agent run / order /
    # backtest message piles up in Redis forever and the run sits in
    # status=queued. In dev we run one worker across all queues; in prod
    # you'd run separate workers per queue for isolation.
    (
        cd "$BACKEND_DIR"
        DJANGO_SETTINGS_MODULE=config.settings.dev \
            "$VENV_CELERY" -A config worker -l info \
            -Q celery,agents,orders,backtests \
            >"$ROOT/logs/celery.log" 2>&1 &
        echo $! >"$ROOT/logs/celery.pid"
    )

    echo "[3/4] starting Celery beat"
    # Beat schedules periodic tasks: broker refresh (30s), order outbox poll
    # (1s), portfolio snapshots (60s), agent-run expiration (5min), broker
    # snapshot prune (6h). Without beat, no periodic tasks fire — the UI
    # falls back to manual /refresh clicks only.
    (
        cd "$BACKEND_DIR"
        DJANGO_SETTINGS_MODULE=config.settings.dev \
            "$VENV_CELERY" -A config beat -l info \
            >"$ROOT/logs/beat.log" 2>&1 &
        echo $! >"$ROOT/logs/beat.pid"
    )
else
    echo "[2-3/4] skipping Celery worker + beat (--no-celery)"
fi

# --- Vite on :5173 --------------------------------------------------------
if [[ "$WITH_FRONT" == "1" ]]; then
    echo "[4/4] starting Vite         → http://localhost:5173"
    (
        cd "$FRONTEND_DIR"
        npm run dev >"$ROOT/logs/vite.log" 2>&1 &
        echo $! >"$ROOT/logs/vite.pid"
    )
    if [[ "$WITH_VERIFY" == "1" ]]; then
        # Vite startup is fast (~2s) but `npm run dev` can stall if
        # node_modules is broken. 20s timeout is generous.
        if ! wait_port_listen 5173 "vite" "$ROOT/logs/vite.log" 20; then
            echo "WARN: Vite didn't come up — backend is fine, UI won't load." >&2
        else
            echo "       vite is live on :5173"
        fi
    fi
else
    echo "[4/4] skipping Vite (--no-front)"
fi

echo
echo "All processes started.  Tail logs with:"
echo "    tail -f logs/web.log logs/celery.log logs/beat.log logs/vite.log"
echo
echo "Stop everything with:"
echo "    bash scripts/dev_down.sh"
