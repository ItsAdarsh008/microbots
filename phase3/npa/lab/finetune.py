"""Fine-tune a finished run's weights with a different training recipe (H21: keep the growth, add the repair).

    python lab/finetune.py base_10k --tag all_damage              # damage on every kind, lr 2e-4, 3,000 iterations
    python lab/finetune.py gated3_rms --tag ramp --iters 4000

Writes lab/queue/master/<prio>_<run>__ft_<tag>.json with the run's weights embedded, so any GPU host can run it.
The run must have a params.npz (fleet.py pulls them). The cell, shape and eps stay as in the original run.
"""

import argparse
import base64
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
ALL_DAMAGE = [{"kind": "wipe", "level": [0.2, 0.2], "p": 1.0}, {"kind": "cut", "level": [0.1, 0.3], "p": 0.5},
              {"kind": "drag", "level": [0.2, 0.8], "p": 0.3}, {"kind": "clump", "level": [0.3, 0.7], "p": 0.3}]


def make(run, tag, iters=3000, lr=2e-4, prio=42):
    res, npz = RUNS / run / "result.json", RUNS / run / "params.npz"
    if not (res.exists() and npz.exists()):
        return None
    cfg = copy.deepcopy(json.loads(res.read_text())["config"])
    cfg["name"] = f"{run}__ft_{tag}"
    cfg["init_params_b64"] = base64.b64encode(npz.read_bytes()).decode()
    cfg["train"].update({"iters": iters, "lr": lr, "damage": ALL_DAMAGE})
    path = HERE / "queue" / "master" / f"{prio:02d}_{cfg['name']}.json"
    if path.exists():
        return None
    path.write_text(json.dumps(cfg))
    return cfg["name"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="all_damage")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    (HERE / "queue" / "master").mkdir(parents=True, exist_ok=True)
    print("queued:", [m for m in (make(r, a.tag, a.iters, a.lr) for r in a.runs) if m])
