"""Run queued experiment configs one after another, unattended.

    python lab/queue_runner.py [--gpu 0] [--idle-exit]

Configs are JSON files in lab/queue/pending/. The lowest file name goes first, so prefix names with a
priority number (10_baseline.json before 20_width.json). A runner claims a config by moving it to
running/ (an atomic rename, so several runners on several GPUs can share one folder), runs
lab/run.py on it, and moves it to done/ or, after RETRIES failed attempts, to failed/.

It keeps going until you create the file lab/STOP (or press ctrl-C), waiting for new configs when
the queue is empty. It also stops if the disk is nearly full.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE / "queue"
RETRIES = 2
MIN_FREE_GB = 20
GRACE_MINUTES = 15          # time allowed beyond the config's own max_minutes (compile + scoring)


def claim():
    """Atomically take the first pending config, or None."""
    for path in sorted((QUEUE / "pending").glob("*.json")):
        target = QUEUE / "running" / path.name
        try:
            os.rename(path, target)
            return target
        except FileNotFoundError:      # another runner got it first
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="CUDA device index for this runner")
    ap.add_argument("--idle-exit", action="store_true", help="exit when the queue is empty instead of waiting")
    args = ap.parse_args()
    for sub in ("pending", "running", "done", "failed"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"runner up (gpu {args.gpu}); create {HERE / 'STOP'} to stop it", flush=True)

    while True:
        if (HERE / "STOP").exists():
            print("STOP file found; exiting", flush=True)
            return
        if shutil.disk_usage(HERE).free / 1e9 < MIN_FREE_GB:
            print(f"less than {MIN_FREE_GB} GB free; exiting", flush=True)
            return
        job = claim()
        if job is None:
            if args.idle_exit:
                return
            time.sleep(30)
            continue
        cfg = json.loads(job.read_text())
        name = cfg["name"]
        limit = ((cfg.get("train", {}).get("max_minutes") or 120) + GRACE_MINUTES) * 60
        out = HERE.parent / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        ok = False
        for attempt in range(1, RETRIES + 2):
            print(f"{time.strftime('%H:%M:%S')} start {name} (attempt {attempt})", flush=True)
            t0 = time.time()
            try:
                with open(out / "stdout.log", "a") as log:
                    code = subprocess.run([sys.executable, str(HERE / "run.py"), str(job)], env=env, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=limit).returncode
            except subprocess.TimeoutExpired:
                code = -1
                print(f"  {name} timed out after {limit / 60:.0f} min", flush=True)
            print(f"{time.strftime('%H:%M:%S')} {name} exit {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
            if code == 0:
                ok = True
                break
        os.rename(job, QUEUE / ("done" if ok else "failed") / job.name)


if __name__ == "__main__":
    main()
