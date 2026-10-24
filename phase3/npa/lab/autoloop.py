"""Invent the next experiments from the results so far, so the GPUs never run out of work.

Three moves, in this order of priority:
  replicate  rerun the best few configs with other seeds (a gain smaller than the seed spread is not a gain)
  transfer   run the best lizard configs on the other shapes
  mutate     take the current best config for a shape and change one knob
Nothing here judges the science; it just proposes. Hypothesis-driven experiments are written by hand
(plan.py). generate() never repeats a config it has seen anywhere (queued, running, done, failed).
"""

import copy
import hashlib
import json
import random
from pathlib import Path

import shapes

LAB = Path(__file__).resolve().parent
RUNS = LAB.parent / "runs"
ALL_DAMAGE = [{"kind": "wipe", "level": [0.2, 0.2], "p": 1.0}, {"kind": "cut", "level": [0.1, 0.3], "p": 0.5},
              {"kind": "drag", "level": [0.2, 0.8], "p": 0.3}, {"kind": "clump", "level": [0.3, 0.7], "p": 0.3}]

CELL_KNOBS = {"width": [64, 128, 256, 512], "depth": [2, 3, 4], "block": ["plain", "residual", "gated"],
              "norm": ["none", "rms"], "mem": [0, 4, 8, 16], "scales": [[1.0], [1.0, 2.0], [1.0, 1.5, 2.0]]}
TRAIN_KNOBS = {"noise_state": [0.0, 0.01, 0.02, 0.05], "noise_pos": [0.0, 0.03, 0.05, 0.1],
               "steps": [[32, 96], [64, 160], [16, 64]], "lr": [3e-4, 5e-4, 1e-3], "batch": [4, 8, 16],
               "grad": ["per_tensor", 1.0, 5.0],
               "damage": [ALL_DAMAGE[:1], ALL_DAMAGE[:2], ALL_DAMAGE[:3], ALL_DAMAGE,
                          [{"kind": "wipe", "level": [0.2, 0.6], "p": 1.0}] + ALL_DAMAGE[1:]]}


def key(cfg):
    """Identity of a config ignoring its name and seed."""
    c = {k: v for k, v in cfg.items() if k not in ("name", "seed")}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def known_configs():
    seen = {}
    for p in (LAB / "queue").rglob("*.json"):
        try:
            c = json.loads(p.read_text())
            seen[(key(c), c.get("seed", 0))] = c["name"]
        except (ValueError, KeyError):
            pass
    for p in RUNS.glob("*/result.json"):
        try:
            c = json.loads(p.read_text())["config"]
            seen[(key(c), c.get("seed", 0))] = c["name"]
        except (ValueError, KeyError):
            pass
    return seen


def results():
    rows = []
    for p in RUNS.glob("*/result.json"):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if "config" in d and d["summary"].get("finite") and "cell" in d["config"] and "train" in d["config"] \
                and "shapes" not in d["config"] and d["config"]["train"].get("mode", "bptt") in ("bptt", "trunc"):
            rows.append(d)
    return rows


def generate(n=8, rng=None):
    """Return up to n new configs as (priority, config) pairs."""
    rng = rng or random.Random()
    rows, seen = results(), known_configs()
    taken = set(seen.values())
    out = []

    def add(prio, cfg):
        if len(out) >= n or (key(cfg), cfg.get("seed", 0)) in seen or cfg["name"] in taken:
            return
        seen[(key(cfg), cfg.get("seed", 0))] = cfg["name"]
        taken.add(cfg["name"])
        out.append((prio, cfg))

    by_shape = {}
    for d in rows:
        by_shape.setdefault(d.get("shape", "lizard"), []).append(d)
    for shape in by_shape:
        by_shape[shape].sort(key=lambda d: d["summary"]["score"])

    # promote: confirm the best screening runs (by score, and by hard-damage excess) at 10,000 iterations
    screened = [d for d in by_shape.get("lizard", []) if d["config"]["train"].get("iters", 0) <= 3000 and d["config"].get("seed", 0) == 0]
    hard = lambda d: sum(d["summary"]["per_kind_excess"][k] for k in ("cut", "drag", "clump"))
    best_ctrl = min((d["summary"]["control_loss"] for d in screened), default=0.0)
    grew = [d for d in screened if d["summary"]["control_loss"] <= 1.6 * best_ctrl]      # ignore runs that never grew a lizard
    for d in sorted(screened, key=lambda d: d["summary"]["score"])[:4] + sorted(grew, key=hard)[:4]:
        c = copy.deepcopy(d["config"])
        c["train"]["iters"], c["name"] = 10000, f"{d['name']}_10k"
        add(15, c)
    # replicate the best few lizard configs
    for d in [d for d in by_shape.get("lizard", []) if d["config"].get("seed", 0) == 0][:3]:
        for seed in (1, 2):
            c = copy.deepcopy(d["config"])
            c["seed"], c["name"] = seed, f"{d['name'].split('_s')[0]}_s{seed}"
            add(12, c)
    # transfer the best two lizard configs to the other shapes
    for d in [d for d in by_shape.get("lizard", []) if d["config"].get("seed", 0) == 0][:2]:
        for shape in shapes.SUITE[1:]:
            c = copy.deepcopy(d["config"])
            c["shape"], c["name"] = shape, f"{d['name']}_on_{shape}"
            add(48, c)
    # mutate the best config of each shape
    for shape, ds in by_shape.items():
        for d in ds[:2]:
            for _ in range(4):
                c = copy.deepcopy(d["config"])
                c["seed"] = 0
                pool = dict(TRAIN_KNOBS) if c["cell"].get("kind", "mlp") != "mlp" else {**TRAIN_KNOBS, **CELL_KNOBS}
                knob = rng.choice(sorted(pool))
                value = rng.choice([v for v in pool[knob] if v != (c["train"].get(knob) if knob in TRAIN_KNOBS else c["cell"].get(knob))])
                (c["train"] if knob in TRAIN_KNOBS else c["cell"])[knob] = value
                c["name"] = f"mut_{shape}_{key(c)[:6]}"
                add(60, c)
    return out
