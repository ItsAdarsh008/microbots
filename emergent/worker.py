"""Per-machine job queue: ~/emergent/queue/{pending,running,done,failed}/<job>.json; several workers share one GPU.
Claim = atomic os.rename pending -> running. A job resumes from runs/<name>/ckpt.pt if it was interrupted.
    python worker.py --slot 0      # loops forever, polling pending every 30 s
"""
import os, sys, time, subprocess
Q = os.path.expanduser("~/emergent/queue")
for d in ("pending", "running", "done", "failed"):
    os.makedirs(f"{Q}/{d}", exist_ok=True)
slot = sys.argv[sys.argv.index("--slot") + 1] if "--slot" in sys.argv else "0"
while True:
    jobs = sorted(os.listdir(f"{Q}/pending"))
    if not jobs:
        time.sleep(30)
        continue
    j = jobs[0]
    try:
        os.rename(f"{Q}/pending/{j}", f"{Q}/running/{j}")
    except FileNotFoundError:
        continue
    import json
    try:
        script = "rl.py" if json.load(open(f"{Q}/running/{j}")).get("algo") == "rl" else "train.py"
    except Exception:
        script = "train.py"
    r = subprocess.run([sys.executable, script, f"{Q}/running/{j}"], cwd=os.path.expanduser("~/emergent"),
                       stdout=open(f"{Q}/running/{j}.out", "w"), stderr=subprocess.STDOUT)
    dest = "done" if r.returncode == 0 else "failed"
    os.rename(f"{Q}/running/{j}", f"{Q}/{dest}/{j}")
    os.rename(f"{Q}/running/{j}.out", f"{Q}/{dest}/{j}.out")
