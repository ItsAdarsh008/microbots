#!/bin/bash
# Keeps the fleet healthy and the dashboard fresh even when no Claude session is awake.
#   tmux new-session -d -s watchdog 'bash phase3/npa/lab/watchdog.sh'
# The dispatcher (lab/fleet.py) restarts this session if it dies, and health.py restarts the dispatcher: they watch each other. Steps have timeouts so a hung ssh cannot freeze the loop.
# Every 10 minutes: health check with auto-restart of the dispatcher. Every 30 minutes: rebuild and deploy the dashboard (no new clips).
cd "$(dirname "$0")/.." || exit 1
PY=../../.venv/bin/python
n=0
while [ ! -f lab/STOP_WATCHDOG ]; do
  { date; timeout 300 $PY lab/health.py --fix; } >> lab/watchdog.log 2>&1
  if [ $((n % 3)) -eq 0 ]; then { timeout 600 $PY lab/dashboard.py --deploy; } >> lab/watchdog.log 2>&1; fi
  n=$((n + 1))
  sleep 600
done
