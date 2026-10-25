"""Keep every GPU busy: hand out experiments, restart runners, sync code, pull results back.

    python lab/fleet.py            # runs until you create lab/STOP_FLEET (run it inside tmux)

Each cycle, for every host: sync the code, make sure its queue runners are alive, top its queue up from
the master queue (lab/queue/master/, lowest file name first) and pull finished runs into runs/.
When the master queue runs low, autoloop.generate() invents more, so GPUs are never left idle.
Hosts that are unreachable are skipped and retried next cycle.
"""

import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import autoloop  # noqa: E402

NPA = HERE.parent
QUEUE = HERE / "queue"
REMOTE = "~/npa-lab/phase3/npa"
HOSTS = {   # name: ip; the GPUs to drive; runners sharing each GPU; (optional) a tmux session that must finish first
    "a10": {"ip": "129.80.219.24", "gpus": [0], "per_gpu": 1, "wait_session": "baseline", "legacy_names": True},
    "a100a": {"ip": "129.213.88.92", "gpus": [0], "per_gpu": 2, "legacy_names": True},
    "a100b": {"ip": "150.136.214.12", "gpus": [0], "per_gpu": 2, "legacy_names": True},
    "a100c": {"ip": "158.101.123.118", "gpus": [0], "per_gpu": 2, "legacy_names": True},
    "h100": {"ip": "192.222.52.190", "gpus": [0, 1], "per_gpu": 3},
}
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
CYCLE_SECONDS = 60
LOW_WATER = 8        # generate more experiments when the master queue has fewer than this


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(HERE / "fleet.log", "a") as f:
        f.write(line + "\n")


def ssh(host, cmd, timeout=60):
    try:
        r = subprocess.run(SSH + [f"ubuntu@{HOSTS[host]['ip']}", cmd], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, ""


def sync_code(host):
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), "--exclude=__pycache__", "--exclude=lab/queue", "--exclude=runs",
                    "--exclude=results", "--exclude=web", "--exclude=lab/results.jsonl", "--exclude=lab/fleet*",
                    f"{NPA}/", f"ubuntu@{HOSTS[host]['ip']}:{REMOTE}/"], capture_output=True, timeout=120)


def pull_results(host):
    (NPA / "runs").mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), "--include=*/", "--include=result.json", "--include=train.log", "--include=params.npz",
                    "--exclude=*", f"ubuntu@{HOSTS[host]['ip']}:{REMOTE}/runs/", f"{NPA}/runs/"],
                   capture_output=True, timeout=180)


def service(host):
    h = HOSTS[host]
    code, out = ssh(host, f"cd {REMOTE}/lab && mkdir -p queue/pending queue/running queue/done queue/failed && "
                          f"echo $(ls queue/pending | wc -l) $(ls queue/running | wc -l) && tmux ls 2>/dev/null | cut -d: -f1 | tr '\\n' ' '")
    if code != 0 or not out:
        log(f"{host}: unreachable, skipping this cycle")
        return None
    first, _, sessions = out.partition("\n")
    pending, running = map(int, first.split()[:2])
    sessions = sessions.split()
    if h.get("wait_session") in sessions:
        return {"host": host, "pending": pending, "running": running, "runners": 0}
    started, total = 0, len(h["gpus"]) * h["per_gpu"]
    for g in h["gpus"]:
        for i in range(h["per_gpu"]):
            name = f"runner{i}" if h.get("legacy_names") else f"runner{g}_{i}"
            if name not in sessions:
                ssh(host, f"cd {REMOTE} && tmux new-session -d -s {name} '~/npa-lab/.venv/bin/python lab/queue_runner.py --gpu {g} >> {name}.log 2>&1'")
                started += 1
    if started:
        log(f"{host}: started {started} runner(s)")
    return {"host": host, "pending": pending, "running": running, "runners": total}


def top_up(st):
    host = st["host"]
    want = st["runners"] + 1 - st["pending"]
    sent = []
    for path in sorted((QUEUE / "master").glob("*.json"))[:max(want, 0)]:
        dest = QUEUE / "dispatched" / host
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(path), f"ubuntu@{HOSTS[host]['ip']}:{REMOTE}/lab/queue/pending/"],
                           capture_output=True, timeout=60)
        if r.returncode == 0:
            os.rename(path, dest / path.name)
            sent.append(path.stem)
    if sent:
        log(f"{host}: queued {', '.join(sent)}")


def ensure_watchdog():
    """The watchdog (tmux 'watchdog', lab/watchdog.sh) and this dispatcher watch each other: if the watchdog's session is gone, start it."""
    if subprocess.run(["tmux", "has-session", "-t", "watchdog"], capture_output=True).returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", "watchdog", f"bash {HERE / 'watchdog.sh'}"], capture_output=True)
        log("watchdog session was missing: restarted it")


def main():
    for sub in ("master", "dispatched"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)
    pid = HERE / "fleet.pid"
    if pid.exists() and Path(f"/proc/{pid.read_text().strip()}").exists():
        raise SystemExit("another fleet.py is already running")
    pid.write_text(str(os.getpid()))
    log(f"fleet up: {', '.join(f'{h} x{len(v['gpus']) * v['per_gpu']}' for h, v in HOSTS.items())}")
    cycle = 0
    while not (HERE / "STOP_FLEET").exists():
        cycle += 1
        try:
            ensure_watchdog()
            if len(list((QUEUE / "master").glob("*.json"))) < LOW_WATER:
                new = autoloop.generate(n=10, rng=random.Random())
                for prio, cfg in new:
                    (QUEUE / "master" / f"{prio:02d}_{cfg['name']}.json").write_text(json.dumps(cfg, indent=1))
                if new:
                    log(f"generated {len(new)} experiments: {', '.join(c['name'] for _, c in new)}")
            with ThreadPoolExecutor(len(HOSTS)) as pool:
                if cycle % 10 == 1:
                    list(pool.map(sync_code, HOSTS))
                states = [s for s in pool.map(service, HOSTS) if s]
                for st in states:
                    top_up(st)
                list(pool.map(pull_results, HOSTS))
            busy = sum(s["running"] for s in states)
            log(f"cycle {cycle}: {busy} running, {sum(s['pending'] for s in states)} pending on hosts, "
                f"{len(list((QUEUE / 'master').glob('*.json')))} in master, {len(list((NPA / 'runs').glob('*/result.json')))} results")
        except Exception as e:      # keep going no matter what
            log(f"cycle {cycle} error: {type(e).__name__}: {e}")
        time.sleep(CYCLE_SECONDS)
    pid.unlink(missing_ok=True)
    log("STOP_FLEET found; exiting")


if __name__ == "__main__":
    main()
