#!/bin/bash
# On a box: wait for the dataset, then start N workers (default 6) that are not already running.
cd ~/emergent; mkdir -p logs queue/pending
until [ -f data/world128.npz ]; do sleep 10; done
N=${1:-6}
for i in $(seq 0 $((N - 1))); do
  pgrep -f "worker.py --slot $i\$" > /dev/null || { setsid nohup python3 worker.py --slot $i > logs/worker$i.log 2>&1 < /dev/null & }
  sleep 3
done
