#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AgenticTrading — start every local dev process in one shot.
#
#   :8000   Django v2 stack      (config.settings.dev, Postgres via DATABASE_URL)
#   :8001   Django legacy bridge (config.settings.legacy, sqlite at repo root)
#   :5173   Vite (React SPA — Vite proxy splits /api/v1/legacy/ to :8001)
#
# Usage:
#   bash scripts/dev_up.sh              # start everything
#   bash scripts/dev_up.sh --no-front   # backends only
#
# Each process writes to logs/<name>.log and its PID to logs/<name>.pid so
# you can `tail -f logs/v2.log` or `kill $(cat logs/v2.pid)` to manage them.
# Ctrl-C (or re-running the script) cleans up the prior processes.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
SQLITE="$ROOT/db.sqlite3"

# --- sanity ----------------------------------------------------------------
if [[ ! -f "$SQLITE" ]]; then
    echo "ERROR: $SQLITE not found — the legacy bridge needs this file." >&2
    exit 1
fi
if [[ ! -f "$BACKEND_DIR/manage.py" ]]; then
    echo "ERROR: $BACKEND_DIR/manage.py not found." >&2
    exit 1
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
stop_pid logs/v2.pid
stop_pid logs/legacy.pid
stop_pid logs/vite.pid

# --- backend v2 on :8000 --------------------------------------------------
echo "[1/3] starting v2 backend  → http://localhost:8000"
(
    cd "$BACKEND_DIR"
    DJANGO_SETTINGS_MODULE=config.settings.dev \
        python manage.py runserver 0.0.0.0:8000 --noreload \
        >"$ROOT/logs/v2.log" 2>&1 &
    echo $! >"$ROOT/logs/v2.pid"
)

# --- backend legacy on :8001 ----------------------------------------------
echo "[2/3] starting legacy bridge → http://localhost:8001 (sqlite: $SQLITE)"
(
    cd "$BACKEND_DIR"
    DJANGO_SETTINGS_MODULE=config.settings.legacy \
        python manage.py runserver 0.0.0.0:8001 --noreload \
        >"$ROOT/logs/legacy.log" 2>&1 &
    echo $! >"$ROOT/logs/legacy.pid"
)

# --- vite on :5173 --------------------------------------------------------
if [[ "${1:-}" != "--no-front" ]]; then
    echo "[3/3] starting Vite        → http://localhost:5173"
    (
        cd "$FRONTEND_DIR"
        npm run dev >"$ROOT/logs/vite.log" 2>&1 &
        echo $! >"$ROOT/logs/vite.pid"
    )
fi

echo
echo "All processes started.  Logs in $ROOT/logs/ — tail them with:"
echo "    tail -f logs/v2.log logs/legacy.log logs/vite.log"
echo
echo "Stop everything with:"
echo "    bash scripts/dev_down.sh"
