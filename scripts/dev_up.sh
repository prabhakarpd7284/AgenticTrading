#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AlphaDesk — start the three local dev processes in one shot.
#
#   :8000   Django ASGI         (REST + WebSocket + broker WS)
#   :5173   Vite (React SPA)
#   —       Celery worker       (order outbox, agent runs, snapshots)
#
# Postgres + Redis live in docker-compose.dev.yml — bring them up separately:
#     docker compose -f docker-compose.dev.yml up -d
#
# Usage:
#     bash scripts/dev_up.sh                 # start everything
#     bash scripts/dev_up.sh --no-front      # backend + worker only
#     bash scripts/dev_up.sh --no-celery     # web + ui only (no async tasks)
#
# Each process writes to logs/<name>.log and its PID to logs/<name>.pid so
# you can `tail -f logs/web.log` or `kill $(cat logs/web.pid)` manually.
# Re-running the script (or scripts/dev_down.sh) cleans up the prior run.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
VENV_PY="$BACKEND_DIR/.venv/bin/python"
VENV_CELERY="$BACKEND_DIR/.venv/bin/celery"

# --- flags -----------------------------------------------------------------
WITH_FRONT=1
WITH_CELERY=1
for arg in "$@"; do
    case "$arg" in
        --no-front)  WITH_FRONT=0 ;;
        --no-celery) WITH_CELERY=0 ;;
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

# --- kill anything left over from a previous run --------------------------
stop_pid() {
    local pidfile="$1"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile" || true)"
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  stopping pid=$pid ($pidfile)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}
stop_pid logs/web.pid
stop_pid logs/celery.pid
stop_pid logs/vite.pid

# --- Django ASGI on :8000 -------------------------------------------------
echo "[1/3] starting Django ASGI  → http://localhost:8000"
(
    cd "$BACKEND_DIR"
    DJANGO_SETTINGS_MODULE=config.settings.dev \
        "$VENV_PY" manage.py runserver 0.0.0.0:8000 --noreload \
        >"$ROOT/logs/web.log" 2>&1 &
    echo $! >"$ROOT/logs/web.pid"
)

# --- Celery worker --------------------------------------------------------
if [[ "$WITH_CELERY" == "1" ]]; then
    echo "[2/3] starting Celery worker"
    (
        cd "$BACKEND_DIR"
        DJANGO_SETTINGS_MODULE=config.settings.dev \
            "$VENV_CELERY" -A config worker -l info \
            >"$ROOT/logs/celery.log" 2>&1 &
        echo $! >"$ROOT/logs/celery.pid"
    )
else
    echo "[2/3] skipping Celery (--no-celery)"
fi

# --- Vite on :5173 --------------------------------------------------------
if [[ "$WITH_FRONT" == "1" ]]; then
    echo "[3/3] starting Vite         → http://localhost:5173"
    (
        cd "$FRONTEND_DIR"
        npm run dev >"$ROOT/logs/vite.log" 2>&1 &
        echo $! >"$ROOT/logs/vite.pid"
    )
else
    echo "[3/3] skipping Vite (--no-front)"
fi

echo
echo "All processes started.  Tail logs with:"
echo "    tail -f logs/web.log logs/celery.log logs/vite.log"
echo
echo "Stop everything with:"
echo "    bash scripts/dev_down.sh"
