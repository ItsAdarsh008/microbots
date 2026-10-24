"""Damage the grown lizard in different ways and watch whether the swarm repairs itself.

    python phase3/npa/repair.py phase3/npa/web/weights/lizard_paper.js
    python phase3/npa/repair.py WEIGHTS.js --scenarios erase_head cut_tail --gif media/npa_repair.gif

Each scenario grows the lizard from the seed, lets it settle for SETTLE steps, damages it once, then
runs RECOVER more steps. The run uses web/headless.js (the demo's engine, which matches the JAX code
to float precision, and is much faster). Every snapshot is scored with the loss terms and IoU from
target.py, so the numbers are comparable with baseline.py.

  erase_head  wipe the state of the head (the damage training uses; positions are untouched)
  cut_tail    delete the particles at the tip of the tail
  cut_limb    delete the particles of a hand or foot
  drag_limb   rip a hand or foot's particles away from the body (positions are displaced)
  control     no damage: does the finished lizard stay put?

Cutting removes particles for good and the rule has no way to make new ones, so a cut can only be
repaired by the remaining particles spreading out to refill the shape.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageDraw

import target as tgt
import viz

HERE = Path(__file__).resolve().parent
SETTLE = 200         # steps to grow and settle before the damage (a trained rollout is 32-96)
RECOVER = 240        # steps to watch afterwards
SNAP_EVERY = 4

SCENARIOS = {
    "erase_head": {"label": "wipe the head", "events": [{"kind": "erase", "where": "head", "r": 0.30}]},
    "cut_tail": {"label": "cut off the tail tip", "events": [{"kind": "cut", "where": "tail", "r": 0.25}]},
    "cut_limb": {"label": "cut off a foot", "events": [{"kind": "cut", "where": "left", "r": 0.20}]},
    "drag_limb": {"label": "rip a foot off", "events": [{"kind": "drag", "where": "left", "r": 0.20, "dx": -0.30, "dy": 0.0}]},
    "control": {"label": "no damage", "events": []},
}


def run_scenarios(weights_js, names, particles, seed):
    """Run the chosen scenarios in parallel, one node process each. Returns {name: (meta, frames)}."""
    tmp = Path(tempfile.mkdtemp())
    procs = {}
    for name in names:
        events = [{**e, "at": SETTLE} for e in SCENARIOS[name]["events"]]
        cmd = ["node", str(HERE / "web" / "headless.js"), weights_js, str(tmp / name), "--particles", str(particles),
               "--steps", str(SETTLE + RECOVER), "--every", str(SNAP_EVERY), "--seed", str(seed),
               "--events", json.dumps(events)]
        procs[name] = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    out = {}
    for name, proc in procs.items():
        print(f"{name}: {proc.communicate()[0].strip()}", flush=True)
        meta = json.loads((tmp / name / "meta.json").read_text())
        raw = np.fromfile(tmp / name / "frames.bin", dtype=np.float32)
        frames, at = [], 0
        for f in meta["frames"]:
            n = f["n"]
            x = raw[at: at + n * 2].reshape(n, 2)
            s = raw[at + n * 2: at + n * 18].reshape(n, 16)
            at += n * 18
            frames.append((f, x.copy(), s.copy()))
        out[name] = (meta, frames)
    return out


def score(frames, evaluate):
    rows = []
    for f, x, s in frames:
        m = {k: float(v) for k, v in evaluate(x, s).items()}
        rows.append({"step": f["step"], "tag": f["tag"], "n": f["n"], **m})
    return rows


def at_step(rows, step, tag=None):
    """The last snapshot at or before `step` (preferring the one tagged `tag` at exactly that step)."""
    if tag:
        for r in rows:
            if r["step"] == step and r["tag"] == tag:
                return r
    return max((r for r in rows if r["step"] <= step and r["tag"] != "damaged"), key=lambda r: r["step"])


def caption(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 16], fill=(18, 20, 28))
    d.text((5, 3), text, fill=(235, 238, 245))
    return img


def make_gif(path, names, results, size=240):
    """One panel per scenario, all stepping together. A ring marks the damage for the first steps after it."""
    panels = []
    for name in names:
        meta, frames = results[name]
        panels.append([(f, x, s) for f, x, s in frames])
    count = min(len(p) for p in panels)
    images = []
    for k in range(count):
        row = []
        for name, p in zip(names, panels):
            f, x, s = p[k]
            mark = None
            for j in range(k, max(k - 6, -1), -1):          # keep the ring for a few snapshots
                if p[j][0]["tag"] == "damaged":
                    mark = p[j][0]["mark"]
            img = viz.draw(x, s, size=size, mark=tuple(mark) if mark else None, radius=max(1.0, size / 200))
            row.append(caption(img, f"{SCENARIOS[name]['label']}  -  step {f['step']}  ({f['n']} particles)"))
        images.append(viz.strip(row, gap=2))
    images = [im.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for im in images]
    # pause on the last frame and hold the damage moment a little longer
    durations = [60] * len(images)
    durations[-1] = 1500
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print("wrote", path, f"({Path(path).stat().st_size / 1e6:.1f} MB, {len(images)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights_js")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--particles", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="repair")
    ap.add_argument("--gif", default=None, help="write a GIF with one panel per scenario")
    args = ap.parse_args()

    results = run_scenarios(args.weights_js, args.scenarios, args.particles, args.seed)
    evaluate = jax.jit(tgt.make_eval(tgt.load_target()))
    summary, strips = {}, []
    for name in args.scenarios:
        meta, frames = results[name]
        rows = score(frames, evaluate)
        before = at_step(rows, SETTLE - 1)
        entry = {"label": SCENARIOS[name]["label"], "particles": args.particles, "settle": SETTLE, "recover": RECOVER,
                 "removed": sum(e.get("removed", 0) for e in meta["events"]), "before": before, "curve": rows}
        if SCENARIOS[name]["events"]:
            entry["just_after"] = at_step(rows, SETTLE, "damaged")
            entry["after"] = {str(d): at_step(rows, SETTLE + d) for d in (20, 60, 120, RECOVER)}
        summary[name] = entry
        print(f"\n{name} ({SCENARIOS[name]['label']}), {entry['removed']} particles removed")
        shown = [("before", before)] + ([("just after", entry["just_after"])] if "just_after" in entry else []) \
            + [(f"+{d} steps", r) for d, r in ([(int(k), v) for k, v in entry.get("after", {}).items()] if entry.get("after") else
                                               [(RECOVER, at_step(rows, SETTLE + RECOVER))])]
        for label, r in shown:
            print(f"  {label:11s} step {r['step']:3d}  n {r['n']}  iou {r['iou']:.3f}  density {r['density']:.3f}  colour {r['colour']:.3f}")

        # the figure: before, the moment of damage, then recovery
        picks = [f for f in frames if f[0]["step"] == SETTLE - 4 and f[0]["tag"] == "run"] \
            + [f for f in frames if f[0]["tag"] == "damaged"] \
            + [f for f in frames if f[0]["tag"] == "run" and f[0]["step"] in (SETTLE + 20, SETTLE + 60, SETTLE + 120, SETTLE + RECOVER)]
        imgs = [caption(viz.draw(x, s, size=280, mark=tuple(f["mark"]) if f["mark"] else None), f"step {f['step']}")
                for f, x, s in picks]
        strips.append(viz.strip(imgs))

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(summary, indent=1))
    for name, strip in zip(args.scenarios, strips):
        strip.save(out / f"{args.tag}_{name}.png")
    if args.gif:
        make_gif(args.gif, [n for n in args.scenarios if n != "control"], results)


if __name__ == "__main__":
    main()
