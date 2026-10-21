#!/usr/bin/env bash
# Historical-data backfill, runnable without the rest of AlphaDesk.
#
# Needs only the two Docker containers (Postgres :5444 for storage, Redis :6380
# for the shared Angel rate limiter) — not Daphne, Celery, beat or Vite. Runs
# detached at low CPU/IO priority, so you can close the terminal (or stop the
# app stack) and it keeps going. Both steps are resumable: re-running only
# fetches what is missing.
#
#   1. backfill_bhavcopy  — daily candles, every NSE stock (NSE archive)
#   2. backfill_candles   — 1-minute candles, NIFTY 50 + indices (Angel One);
#                           --universe fno for all ~230 F&O stocks
#
# Usage:
#   bash scripts/backfill_data.sh start [--months N] [--universe nifty50|fno]
#   bash scripts/backfill_data.sh status
#   bash scripts/backfill_data.sh log                  # follow progress
#   bash scripts/backfill_data.sh stop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
PY="$BACKEND/.venv/bin/python"
LOG="$ROOT/logs/backfill.log"
PIDFILE="$ROOT/logs/backfill.pid"
mkdir -p "$ROOT/logs"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-}" in
  start)
    shift
    MONTHS=6
    UNIVERSE=nifty50
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --months) MONTHS="$2"; shift 2 ;;
        --universe) UNIVERSE="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
      esac
    done
    if running; then
      echo "backfill already running (pid $(cat "$PIDFILE")) — see: $0 log"; exit 0
    fi
    [[ -x "$PY" ]] || { echo "missing $PY — run 'cd backend && uv sync'" >&2; exit 1; }

    echo "ensuring Postgres + Redis are up..."
    docker compose -f "$ROOT/docker-compose.dev.yml" up -d postgres redis >/dev/null
    for _ in $(seq 1 30); do
      docker compose -f "$ROOT/docker-compose.dev.yml" ps postgres | grep -q healthy && break
      sleep 2
    done

    # setsid: own session (its pid is the session id), survives the terminal
    # closing. nice/ionice: yield CPU and disk to whatever you are doing.
    (
      cd "$BACKEND"
      export DJANGO_SETTINGS_MODULE=config.settings.dev PYTHONUNBUFFERED=1 PY MONTHS UNIVERSE
      setsid nice -n 10 ionice -c 3 bash -c '
        {
          echo "=== backfill started $(date "+%F %T") (months=$MONTHS, universe=$UNIVERSE) ==="
          echo "--- step 1/2: daily candles (NSE bhavcopy) ---"
          "$PY" manage.py backfill_bhavcopy --months "$MONTHS"
          echo "--- step 2/2: 1-minute candles (Angel One) ---"
          "$PY" manage.py backfill_candles --universe "$UNIVERSE" --interval 1m --months "$MONTHS"
          echo "=== backfill finished $(date "+%F %T") ==="
        } 2>&1 | grep --line-buffered -vE "plugin\.registered|strategy_extra|^\{|^\[I "
      ' >>"$LOG" 2>&1 </dev/null &
      echo $! >"$PIDFILE"
    )
    echo "backfill started (pid $(cat "$PIDFILE")), logging to $LOG"
    echo "follow it with:  bash scripts/backfill_data.sh log"
    ;;

  stop)
    if running; then
      # The job leads its own session (sid == pid); kill that whole session
      # (bash wrapper + manage.py + grep). Refuse if it is not a session
      # leader, so a stale pid can never take out an unrelated session.
      pid="$(cat "$PIDFILE")"
      if [[ "$(ps -o sid= -p "$pid" | tr -d ' ')" == "$pid" ]]; then
        pkill -TERM -s "$pid" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
      rm -f "$PIDFILE"
      echo "backfill stopped — re-run 'start' later to resume where it left off"
    else
      rm -f "$PIDFILE"; echo "backfill is not running"
    fi
    ;;

  status)
    if running; then echo "running (pid $(cat "$PIDFILE"))"; else echo "not running"; fi
    [[ -f "$LOG" ]] && { echo "--- last lines of $LOG ---"; tail -n 5 "$LOG"; }
    ;;

  log)
    tail -n 20 -f "$LOG"
    ;;

  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
