#!/usr/bin/env bash
# Stops the processes started by dev_up.sh (web, celery, vite).
# Postgres + Redis are managed by `docker compose -f docker-compose.dev.yml`
# and are NOT touched — bring them down with `docker compose ... down` if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for name in web celery beat flower vite; do
    pidfile="logs/$name.pid"
    if [[ -f "$pidfile" ]]; then
        pid="$(cat "$pidfile" || true)"
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "stopping $name (pid=$pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# Sweep up celery worker/beat orphans whose pidfile was lost (common after
# IDE force-quits or stale dev_up.sh re-runs).
pkill -f "celery -A config worker" 2>/dev/null || true
pkill -f "celery -A config beat"   2>/dev/null || true
pkill -f "celery -A config flower" 2>/dev/null || true

# Sweep up any orphan web process on :8000 — a manually-started daphne or
# a previous-run runserver whose pidfile is gone. Without this, the next
# dev_up.sh gets a dual-binding on :8000 and requests randomly hit the
# old code.
if command -v lsof >/dev/null 2>&1; then
    web_pids="$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${web_pids:-}" ]]; then
        echo "killing stray web pid(s) on :8000 → $web_pids"
        # shellcheck disable=SC2086
        kill $web_pids 2>/dev/null || true
    fi
fi

echo "done."
