#!/bin/bash
# Operations from the laptop.  HOSTS: the GPU boxes.
#   ./ops.sh sync                 copy code (+ GIT_COMMIT) to every box
#   ./ops.sh queue HOST cfg...    copy configs into HOST's pending queue
#   ./ops.sh workers HOST N       start N detached workers on HOST (skips if already running)
#   ./ops.sh status               GPU use and queue counts per box
#   ./ops.sh collect              pull runs/*/{result.json,log.txt,samples.png,traj.gif} into results/
HOSTS="137.23.44.178 150.136.88.128"
cd "$(dirname "$0")"
SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o BatchMode=yes"
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; sleep 10; done; echo "FAILED after retries: $*" >&2; return 1; }
ssh() { command $SSH "$@"; }
rsync() { command rsync -e "$SSH" "$@"; }
case "$1" in
sync)
  git rev-parse HEAD > GIT_COMMIT
  for h in $HOSTS; do retry rsync -az --exclude __pycache__ --exclude data --exclude runs --exclude results ./ ubuntu@$h:~/emergent/ & done; wait ;;
queue)
  h=$2; shift 2
  retry rsync -az "$@" ubuntu@$h:~/emergent/queue/pending/ ;;
workers)
  retry ssh ubuntu@$2 "cd ~/emergent && mkdir -p queue/pending logs && n=\$(pgrep -fc 'worker.py --slot') ; for i in \$(seq \$n \$(($3 - 1))); do setsid nohup python3 worker.py --slot \$i > logs/worker\$i.log 2>&1 < /dev/null & done; sleep 1; pgrep -fc 'worker.py --slot'" ;;
status)
  for h in $HOSTS; do
    ssh -o ConnectTimeout=8 ubuntu@$h "echo \"== $h \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) workers \$(pgrep -fc 'worker.py --slot') | pending \$(ls ~/emergent/queue/pending 2>/dev/null | wc -l) running \$(ls ~/emergent/queue/running/*.json 2>/dev/null | wc -l) done \$(ls ~/emergent/queue/done/*.json 2>/dev/null | wc -l) failed \$(ls ~/emergent/queue/failed/*.json 2>/dev/null | wc -l)\"; for f in ~/emergent/queue/running/*.json; do [ -e \"\$f\" ] || continue; n=\$(basename \$f .json); echo \"  \$n: \$(grep -h '^it ' ~/emergent/runs/\$n/log.txt 2>/dev/null | tail -1 | cut -c1-150)\"; done" || echo "== $h unreachable"
  done ;;
collect)
  mkdir -p results
  for h in $HOSTS; do
    retry rsync -az --include '*/' --include 'result.json' --include 'log.txt' --include 'samples.png' --include 'traj.gif' --exclude '*' ubuntu@$h:~/emergent/runs/ results/ &
  done; wait ;;
esac
