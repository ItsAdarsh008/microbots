"""Run the Phase 3 (particle NCA) checks and print PASS or FAIL for each.

    python phase3/npa/checks.py

Checks the SPH maths, the properties the paper claims for the rule (it does not care where the
swarm sits, and it is unchanged by rescaling space), that the JavaScript demo agrees with the
JAX code, and that the trainer runs and learns. It also looks over results/ for the baseline
run (grow, wipe, regrow) if you have made one with baseline.py. Takes about 2 minutes.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import npa
import target as tgt
import train
import weights

HERE = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def lattice(side, spacing, jitter, seed=0):
    """A square grid of particles, optionally jostled, so neighbours are uneven."""
    rng = np.random.default_rng(seed)
    g = (np.arange(side) - (side - 1) / 2) * spacing
    xs, ys = np.meshgrid(g, g)
    x = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    return jnp.asarray((x + rng.normal(0, jitter * spacing, x.shape)).astype(np.float32))


EPS = 0.1
key = jax.random.PRNGKey(0)

# 1. The model is the size the paper reports (about 11k parameters).
params = npa.init_params(key, gain=1.0)
count = sum(int(np.prod(p.shape)) for p in params.values())
check("the MLP has about 11k parameters", count == 10880, f"{count}")

# 2. What a particle senses has the right shape and is finite, and a zero state gives zero gradients.
x = lattice(20, 0.04, 0.2)
s = jax.random.normal(key, (x.shape[0], npa.STATE_DIMS))
z = npa.perceive(x, s, EPS, EPS)
check("perception is (N, 66) and finite", z.shape == (400, 66) and bool(jnp.isfinite(z).all()), str(z.shape))
z0 = npa.perceive(x, jnp.zeros_like(s), EPS, EPS)
check("an all-zero state senses no gradient and no smoothed state", float(jnp.abs(z0[:, :-2]).max()) == 0.0)

# 3. Density on an even lattice equals the particle count per area (the kernel integrates to 1).
side, spacing = 40, 0.025
xl = lattice(side, spacing, 0.0)
rho = jax.lax.map(lambda xi: jnp.sum(npa._poly6(jnp.sum((xl - xi) ** 2, -1), EPS)), xl)
inner = (jnp.abs(xl) < (side * spacing / 2 - EPS)).all(axis=1)
expected = 1.0 / spacing**2
err = float(jnp.abs(rho[inner] / expected - 1).max())
check("SPH density on an even lattice is N / area", err < 0.02, f"max error {err:.2%}")

# 4. The corrected gradient recovers a linear field exactly, even with uneven neighbours.
xj = lattice(40, 0.025, 0.3, seed=1)
grad_true = np.array([3.0, -2.0])
field = (xj @ jnp.asarray(grad_true, jnp.float32))[:, None] * jnp.ones((1, npa.STATE_DIMS))
zj = np.asarray(npa.perceive(xj, field, EPS, EPS))
g = zj[:, 32:34]                                  # channel 0's gradient, squashed by log(1 + |g|)
norm = np.linalg.norm(g, axis=1, keepdims=True)
g = np.expm1(norm) * g / np.maximum(norm, 1e-12)  # undo the squashing
inner = np.asarray((jnp.abs(xj) < 0.5 - EPS).all(axis=1))
err = np.abs(g[inner] - grad_true).max() / np.linalg.norm(grad_true)
check("the gradient of a linear field is recovered on a jostled lattice", err < 0.05, f"max error {err:.2%}")

# 5. The rule does not care where the swarm sits, or how the particles are numbered.
x = lattice(15, 0.05, 0.3, seed=2)
s = jax.random.normal(key, (x.shape[0], npa.STATE_DIMS)) * 0.5
on = jnp.ones((x.shape[0], 1))
x1, s1, _ = npa.step(params, x, s, on, EPS, EPS)
x2, s2, _ = npa.step(params, x + jnp.array([0.7, -0.3]), s, on, EPS, EPS)
check("shifting the whole swarm changes nothing but the shift",
      float(jnp.abs((x2 - jnp.array([0.7, -0.3])) - x1).max()) < 1e-4 and float(jnp.abs(s2 - s1).max()) < 1e-4)
perm = jax.random.permutation(key, x.shape[0])
x3, s3, _ = npa.step(params, x[perm], s[perm], on, EPS, EPS)
check("renumbering the particles just renumbers the result",
      float(jnp.abs(x3 - x1[perm]).max()) < 1e-5 and float(jnp.abs(s3 - s1[perm]).max()) < 1e-5)

# 6. The paper's scaling claim: rescale space and EPS by c (keeping the trained radius eps0) and
# the dynamics are unchanged: states evolve identically and moves scale by c.
c = 2.0
x4, s4, _ = npa.step(params, x * c, s, on, EPS * c, EPS)
check("rescaling space and EPS by 2 leaves the dynamics unchanged",
      float(jnp.abs(s4 - s1).max()) < 1e-3 and float(jnp.abs((x4 - x * c) - (x1 - x) * c).max()) < 1e-4,
      f"state diff {float(jnp.abs(s4 - s1).max()):.1e}")

# 7. Damage wipes exactly the particles inside the brush.
xd = lattice(20, 0.05, 0.0)
sd = jnp.ones((400, npa.STATE_DIMS))
after = npa.damage(xd, sd, jnp.zeros(2), 0.2)
inside = np.asarray(jnp.sum(xd**2, -1) < 0.04)
check("damage clears the state inside the radius and nothing outside",
      bool((np.asarray(after)[inside] == 0).all()) and bool((np.asarray(after)[~inside] == 1).all()),
      f"{int(inside.sum())} particles cleared")

# 8. A model written for the web demo gives the same answer in JavaScript as in JAX.
node = shutil.which("node")
if node is None:
    print("[SKIP] JavaScript agrees with JAX (node is not installed)")
else:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(3)
        n = 600
        r, t = np.sqrt(rng.random(n)) * 0.35, rng.random(n) * 2 * np.pi
        xs = np.stack([r * np.cos(t), r * np.sin(t)], -1).astype(np.float32)
        ss = (rng.random((n, 16)) - 0.5).astype(np.float32)
        masks = (rng.random((5, n)) < 0.5).astype(np.uint8)
        jp = npa.init_params(jax.random.PRNGKey(5), gain=0.4)    # big enough that outputs matter, small enough not to amplify rounding
        xa, sa = jnp.asarray(xs), jnp.asarray(ss)
        snaps = []
        for k in range(5):
            xa, sa, _ = npa.step(jp, xa, sa, jnp.asarray(masks[k], jnp.float32)[:, None], EPS, EPS)
            snaps.append((np.asarray(xa), np.asarray(sa)))
        (tmp / "fixture.json").write_text(json.dumps({
            "n": n, "eps": EPS, "x": xs.ravel().tolist(), "s": ss.ravel().tolist(), "masks": masks.tolist(),
            "x1": snaps[0][0].ravel().tolist(), "s1": snaps[0][1].ravel().tolist(),
            "x5": snaps[4][0].ravel().tolist(), "s5": snaps[4][1].ravel().tolist()}))
        weights.to_js(tmp / "w.js", "test", {k: np.asarray(v) for k, v in jp.items()}, EPS, EPS)
        out = subprocess.run([node, str(HERE / "web" / "parity.js"), str(tmp / "fixture.json"), str(tmp / "w.js")],
                             capture_output=True, text=True, check=True).stdout
        gap = json.loads(out)
        worst = max(gap["after1"]["x"], gap["after1"]["s"], gap["after5"]["x"], gap["after5"]["s"])
        check("JavaScript matches JAX over 5 steps", worst < 1e-3,
              f"largest gap {worst:.1e} (positions {gap['after5']['x']:.1e}, states {gap['after5']['s']:.1e})")

# 9. The target has about the 4,096 points the paper uses, and a perfect copy of it scores perfectly.
target = tgt.load_target()
check("the lizard target has about 4,096 points", abs(len(target["pos"]) - 4096) < 200, f"{len(target['pos'])}")
evaluate = tgt.make_eval(target)
perfect = evaluate(jnp.asarray(target["pos"]), jnp.asarray(target["rgb"] - 0.5))   # colour = state + 0.5
check("a swarm sitting exactly on the target scores IoU 1 and near-zero loss",
      float(perfect["iou"]) > 0.99 and float(perfect["density"]) < 1e-3 and float(perfect["colour"]) < 1e-3,
      f"iou {float(perfect['iou']):.3f}, density {float(perfect['density']):.1e}")

# 10. The trainer runs, and it learns: on one fixed batch the loss goes down.
train.STEP_RANGE = (4, 8)
n, batch = 256, 2
eps = train.scaled_eps(n)
loss_fn = tgt.make_loss(target, grid=64)
train_step = train.make_train_step(loss_fn, batch, n, eps, eps, None)
p0 = npa.init_params(jax.random.PRNGKey(7))
opt = train.init_opt(p0)
xb, sb = npa.seed(jax.random.PRNGKey(8), n, radius=2 * eps)
xb, sb = jnp.stack([xb, xb]), jnp.stack([sb, sb])
pp, losses = p0, []
for i in range(12):
    pp, opt, _, _, loss, _ = train_step(pp, opt, jax.random.PRNGKey(9), xb, sb, 6, 3e-3)
    losses.append(float(loss))
check("a training step gives a finite loss and changes the weights",
      all(np.isfinite(losses)) and float(jnp.abs(pp["w2"] - p0["w2"]).max()) > 0)
check("the loss falls over 12 steps on a fixed batch", losses[-1] < losses[0] - 0.05,
      f"{losses[0]:.3f} -> {losses[-1]:.3f}")

# 11. If the baseline run has been done, it should show the lizard growing and regrowing.
base = HERE / "results" / "baseline_N4096.json"
if base.exists():
    log = json.loads(base.read_text())["log"]
    grown = [r for r in log if r["phase"] == "grow"][-1]
    healed = [r for r in log if r["phase"] == "heal"][-1]
    wiped = [r for r in log if r["phase"] == "heal"][0]
    check("baseline: the seed grows into the lizard (IoU above 0.8)", grown["iou"] > 0.8, f"IoU {grown['iou']:.2f}")
    # a state wipe leaves the footprint alone, so IoU cannot see it: the colour loss is what shows the repair
    check("baseline: after the head is wiped its colour comes back (colour loss falls to its old level)",
          healed["colour"] < wiped["colour"] and healed["colour"] <= 1.1 * grown["colour"],
          f"{wiped['colour']:.3f} -> {healed['colour']:.3f}, before the wipe {grown['colour']:.3f}")
else:
    print("[SKIP] baseline results not found (run baseline.py to make them)")

# 12. If the damage experiments have been run (repair.py), the damage the model was trained on should heal.
rep = HERE / "results" / "repair.json"
if rep.exists():
    d = json.loads(rep.read_text())
    ctrl = d["control"]["curve"][-1]                       # the same step in the run with no damage
    wiped = d["erase_head"]
    last = wiped["after"][str(wiped["recover"])]
    check("repair: a wiped head regains its colour (colour loss within 15% of an undamaged lizard)",
          last["colour"] <= 1.15 * ctrl["colour"], f"{wiped['just_after']['colour']:.3f} -> {last['colour']:.3f}, undamaged {ctrl['colour']:.3f}")
    check("repair: a wiped head's shape stays as good as an undamaged lizard's (IoU within 0.02)",
          last["iou"] >= ctrl["iou"] - 0.02, f"{last['iou']:.3f} vs {ctrl['iou']:.3f}")
    # Cutting and dragging were never in training, so these are reported, not asserted.
    for name in ("cut_limb", "drag_limb", "cut_tail"):
        if name in d:
            r = d[name]["after"][str(d[name]["recover"])]
            print(f"[INFO] {d[name]['label']}: IoU {d[name]['just_after']['iou']:.3f} -> {r['iou']:.3f} "
                  f"(undamaged {ctrl['iou']:.3f}); look at results/repair_{name}.png for what actually grew back")
else:
    print("[SKIP] repair results not found (run repair.py to make them)")

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
