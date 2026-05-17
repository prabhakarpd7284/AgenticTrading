#!/usr/bin/env bash
# Stops the processes started by dev_up.sh (web, celery, vite).
# Postgres + Redis are managed by `docker compose -f docker-compose.dev.yml`
# and are NOT touched — bring them down with `docker compose ... down` if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for name in web celery vite; do
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
echo "done."
