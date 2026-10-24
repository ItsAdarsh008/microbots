"""One-shot health check of the experiment fleet. Prints problems; restarts the dispatcher if it died.

    python lab/health.py [--fix]

Checks: dispatcher alive and logging recently; every host reachable; GPUs busy; runners alive; jobs stuck (no log
output for 20 minutes); jobs in failed/; disk space; newly finished runs that stopped early or diverged.
Exit text starts with OK or PROBLEMS so a wake-up can act on the first line.
"""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fleet  # noqa: E402

NPA = HERE.parent
problems, notes = [], []


def ssh(host, cmd):
    return fleet.ssh(host, cmd, timeout=40)


# dispatcher
log = HERE / "fleet.log"
age = time.time() - log.stat().st_mtime if log.exists() else 1e9
alive = subprocess.run("tmux has-session -t fleet 2>/dev/null", shell=True).returncode == 0
if not alive or age > 300:
    problems.append(f"dispatcher not healthy (tmux session {'up' if alive else 'DOWN'}, log idle {age:.0f}s)")
    if "--fix" in sys.argv:
        subprocess.run("tmux kill-session -t fleet 2>/dev/null; rm -f %s/fleet.pid; tmux new-session -d -s fleet "
                       "'cd %s && %s lab/fleet.py > lab/fleet.stdout 2>&1'" % (HERE, NPA, sys.executable), shell=True)
        problems[-1] += " -> restarted"

# hosts
for host, h in fleet.HOSTS.items():
    code, out = ssh(host, f"cd {fleet.REMOTE}/lab && echo $(ls queue/running 2>/dev/null | wc -l) $(ls queue/failed 2>/dev/null | wc -l) "
                          f"$(df -BG --output=avail / | tail -1 | tr -dc 0-9) && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\\n' ' ' "
                          f"&& echo && find ../runs -name train.log -mmin -20 | wc -l && find ../runs -name train.log -mmin +20 -newer queue 2>/dev/null | head -0")
    if code != 0 or not out:
        problems.append(f"{host}: unreachable")
        continue
    lines = out.split("\n")
    running, failed, free = map(int, lines[0].split()[:3])
    util = [int(u) for u in lines[1].split()] if len(lines) > 1 and lines[1].strip() else []
    fresh = int(lines[2]) if len(lines) > 2 and lines[2].strip().isdigit() else -1
    if running and fresh == 0:
        problems.append(f"{host}: {running} running job(s) but no train.log written in 20 min (stuck?)")
    if util and max(util) < 20 and running:
        problems.append(f"{host}: GPU idle ({util}%) with {running} running")
    if not running:
        notes.append(f"{host}: nothing running (queue may be empty)")
    if failed:
        notes.append(f"{host}: {failed} job(s) in failed/")
    if free < 30:
        problems.append(f"{host}: only {free} GB disk free")

# local disk and queue
free_local = subprocess.run("df -BG --output=avail / | tail -1 | tr -dc 0-9", shell=True, capture_output=True, text=True).stdout
if free_local.strip() and int(free_local) < 5:
    problems.append(f"Pi disk almost full ({free_local} GB)")

# new results since last check
state = HERE / "health.state.json"
seen = set(json.loads(state.read_text())) if state.exists() else set()
bad = []
now = set()
for p in glob.glob(str(NPA / "runs/*/result.json")):
    name = Path(p).parent.name
    now.add(name)
    if name in seen:
        continue
    d = json.loads(Path(p).read_text())
    want, got = d["config"].get("train", {}).get("iters"), d.get("iters_done")
    sm = d["summary"]
    if want and got is not None and want > 0 and got < 0.9 * want:
        bad.append(f"{name}: stopped at {got}/{want} iterations ({d.get('blowups')} blow-ups)")
    elif not sm.get("finite") or sm["score"] > 100:
        bad.append(f"{name}: diverged (score {sm['score']:.1f})")
state.write_text(json.dumps(sorted(now)))
if bad:
    problems.append("new bad runs: " + "; ".join(bad[:8]) + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""))

master = len(list((HERE / "queue/master").glob("*.json")))
last = log.read_text().strip().split("\n")[-1] if log.exists() else "(no log)"
print(("PROBLEMS" if problems else "OK") + f" | {len(now)} results, {master} in master queue")
for p in problems:
    print(" !", p)
for n in notes:
    print(" -", n)
print(" last fleet line:", last[:140])
