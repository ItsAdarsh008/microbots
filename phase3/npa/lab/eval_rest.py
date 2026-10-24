"""Run the immutable shape-agnostic evaluator on a rest-memory cell and print the summary.

    python lab/eval_rest.py '{"frame": "distance", "imprint_at": 8}' [tiny|medium|full]

Goes through run.train (iters = 0, nothing to learn) so the real dispatch is exercised, then evaluate_agnostic.run.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import evaluate_agnostic as ea  # noqa: E402
import run  # noqa: E402

TINY = {"warm": 8, "recover": 8, "emoji": ("lizard",), "proc_seeds": (0, 1), "sample_seeds": (1,), "grid": 64, "chunk": 2}
MEDIUM = {"emoji": ("lizard", "donut"), "proc_seeds": tuple(range(6)), "sample_seeds": (910_000,), "grid": 64, "chunk": 8}

if __name__ == "__main__":
    opts = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    mode = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    protocol = {"tiny": TINY, "medium": MEDIUM, "full": {}}[mode]
    cfg = {"name": "rest_eval", "particles": 128, "cell": {"kind": "rest", **opts},
           "train": {"mode": "agnostic", "iters": 0, "batch": 2, "grid": 64}}
    t0 = time.time()
    cell, params, eps, curve, secs = run.train(cfg, log=print)
    res = ea.run(cell, params, eps, eps, 128, cfg, protocol=protocol)
    sm = res["summary"]
    print(f"cell {cfg['cell']}  protocol {mode}  params {cell.param_count(params)}  {time.time() - t0:.0f}s")
    print(f"score {sm['score']:.4f}  do_nothing_score {sm['do_nothing_score']:.4f}  control {sm['control_loss']:.4f}  "
          f"do_nothing_control {sm['do_nothing_control']:.4f}  drift {sm['drift']:.4f}  finite {sm['finite']}")
    print("per-kind score    " + "  ".join(f"{k} {v:.3f}" for k, v in sm["per_kind_score"].items()))
    print("do-nothing score  " + "  ".join(f"{k} {v:.3f}" for k, v in sm["do_nothing_per_kind_score"].items()))
    print("repair_fraction   " + "  ".join(f"{k} {v:+.2f}" for k, v in sm["repair_fraction"].items()))
    print("restoration       " + "  ".join(f"{k} {v:+.2f}" for k, v in sm["restoration"].items()))
    print(f"repair_fraction_easy {sm['repair_fraction_easy']:+.3f}")
