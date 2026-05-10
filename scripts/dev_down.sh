#!/usr/bin/env bash
# Stops the three processes started by dev_up.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for name in v2 legacy vite; do
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
