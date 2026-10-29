"""Per-machine job queue: ~/diffusion/queue/{pending,running,done,failed}/<job>.json.
Claim = atomic os.rename pending -> running. Resume is automatic (train.py continues from ckpt.pt).
    python worker.py            # run until pending is empty
"""
import os, sys, time, subprocess
Q = os.path.expanduser("~/diffusion/queue")
for d in ("pending", "running", "done", "failed"):
    os.makedirs(f"{Q}/{d}", exist_ok=True)
while True:
    jobs = sorted(os.listdir(f"{Q}/pending"))
    if not jobs:
        break
    j = jobs[0]
    try:
        os.rename(f"{Q}/pending/{j}", f"{Q}/running/{j}")
    except FileNotFoundError:
        continue          # another worker claimed it
    r = subprocess.run([sys.executable, "train.py", f"{Q}/running/{j}"], cwd=os.path.expanduser("~/diffusion"),
                       stdout=open(f"{Q}/running/{j}.out", "w"), stderr=subprocess.STDOUT)
    os.rename(f"{Q}/running/{j}", f"{Q}/{'done' if r.returncode == 0 else 'failed'}/{j}")
    os.rename(f"{Q}/running/{j}.out", f"{Q}/{'done' if r.returncode == 0 else 'failed'}/{j}.out")
