#!/bin/bash
# Keeps the fleet healthy and the dashboard fresh even when no Claude session is awake.
#   tmux new-session -d -s watchdog 'bash phase3/npa/lab/watchdog.sh'
# Every 10 minutes: health check with auto-restart of the dispatcher. Every 30 minutes: rebuild and deploy the dashboard (no new clips).
cd "$(dirname "$0")/.." || exit 1
PY=../../.venv/bin/python
n=0
while [ ! -f lab/STOP_WATCHDOG ]; do
  { date; $PY lab/health.py --fix; } >> lab/watchdog.log 2>&1
  if [ $((n % 3)) -eq 0 ]; then { $PY lab/dashboard.py --deploy; } >> lab/watchdog.log 2>&1; fi
  n=$((n + 1))
  sleep 600
done
