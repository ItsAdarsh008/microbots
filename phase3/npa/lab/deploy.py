"""Test trained models under a physical-robot embodiment WITHOUT retraining: the sim-to-real gap.

    python lab/deploy.py base_10k phys              # writes lab/queue/master/<prio>_<run>__phys.json
    python lab/deploy.py --all phys lattice         # every finished run that has params.npz

Makes a config that loads the run's trained weights (embedded in the config, so any GPU host can run it),
wraps the same cell in the embodiment (see embodiment.md) and scores it with the normal evaluator. The result
is named <run>__<embodiment> and shows up in report.py next to the original. Compare a run with its
deployed twin: the drop in score is what the dots hid.
"""

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MASTER = HERE / "queue" / "master"
# eps in world units: about two body diameters (Ø30 mm at 900 mm per unit), so ~13 neighbours in the lizard
PHYS_EPS = 0.066

# the recommended physical parameters (embodiment.md D3-D5, locomotion_findings.md), all in world units, 1 unit = 900 mm
PHYS = {"mode": "phys", "sense_noise": 0.0022, "quant_bits": 8, "dropout": 0.3,        # 2 mm range noise, 8-bit states, line-of-sight losses
        "max_step": 0.055, "min_step": 0.02,                                        # strides of 18 to 50 mm
        "step_len_noise": 0.2, "step_dir_noise": 0.087, "bearing_noise": 0.087}     # +-20% length, +-5 deg heading and bearing
LATTICE = {"mode": "lattice", "sense_noise": 0.0022, "quant_bits": 8, "dropout": 0.3, "hop_success": 0.9}
EMBODIMENTS = {"phys": PHYS, "lattice": LATTICE}


def make(run_name, emb, prio=40):
    res = RUNS / run_name / "result.json"
    npz = RUNS / run_name / "params.npz"
    if not res.exists() or not npz.exists():
        return None
    cfg = json.loads(res.read_text())["config"]
    cfg["name"] = f"{run_name}__{emb}"
    cfg["init_params_b64"] = base64.b64encode(npz.read_bytes()).decode()
    cfg["train"] = {**cfg["train"], "iters": 0}
    cfg["cell"] = {**cfg["cell"], "embody": EMBODIMENTS[emb]}
    cfg["eps"] = cfg.get("eps") or None
    path = MASTER / f"{prio:02d}_{cfg['name']}.json"
    if path.exists():
        return None
    path.write_text(json.dumps(cfg))
    return cfg["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--embodiments", nargs="+", default=["phys"])
    args = ap.parse_args()
    names = args.runs
    if args.all:
        names = [p.name for p in RUNS.iterdir() if (p / "params.npz").exists() and "__" not in p.name]
    MASTER.mkdir(parents=True, exist_ok=True)
    made = [make(r, e) for r in names for e in args.embodiments]
    print("queued:", [m for m in made if m])


if __name__ == "__main__":
    main()
