"""Render a swarm run of the body model as a GIF: one shape, four copies (no damage, cut, clump, noise), grown/warmed up,
damaged at the marked frame, then left to repair itself. Bots are discs of the real body radius, coloured by their state.

    python lab/render_swarm.py ag_axis_mlph_scales2 --shape lizard --out swarm.gif
The body model is the statistical stand-in in lab/embody.py (assumed, motion limits measured in MuJoCo), not a physics simulation.
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import agnostic  # noqa: E402
import cells  # noqa: E402
import damage  # noqa: E402
import npa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--shape", default="lizard")
ap.add_argument("--out", default="swarm.gif")
ap.add_argument("--warm", type=int, default=32)
ap.add_argument("--recover", type=int, default=120)
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--panel", type=int, default=330)
a = ap.parse_args()

d = json.loads((HERE.parent / "runs" / a.run / "result.json").read_text())
cfg = d["config"]
z = np.load(HERE.parent / "runs" / a.run / "params.npz")
params = {k: jnp.asarray(z[k]) for k in z.files if k not in ("eps", "eps0")}
eps = float(z["eps"]); eps0 = float(z["eps0"]) if "eps0" in z.files else eps
n = cfg["particles"]
cell = cells.build(**cfg["cell"])
body_r = (cfg["cell"].get("embody") or {}).get("body_r", 0.0129)

mask, rgb = agnostic.get_shape(a.shape) if a.shape.startswith("proc:") else agnostic.emoji_shape(a.shape)
x0, s0 = agnostic.sample_swarm(mask, rgb, n, 910_000)
x0 = jnp.asarray(x0); s0 = jnp.asarray(s0)
x0 = x0 - x0.mean(0)
mem0 = jnp.zeros((n, cell.mem_dims)); alive0 = jnp.ones(n)

step = jax.jit(lambda x, s, mem, alive, k: _step(x, s, mem, alive, k))


def _step(x, s, mem, alive, k):
    update = jax.random.bernoulli(k, npa.UPDATE_PROB, (n, 1)).astype(x.dtype) * alive[:, None]
    x, s, mem, _ = cell.step(params, x, s, mem, update, eps, eps0, None, n, alive)
    return x, s, mem


step = jax.jit(_step)
dmg = {"none": None, "cut": ("cut", 0.30), "clump": ("clump", 0.7), "noise": ("noise", 0.5)}
rng = jax.random.PRNGKey(7)
frames_state = {k: [] for k in dmg}
kw, kd = jax.random.split(rng)
for name, spec in dmg.items():
    x, s, mem, alive = x0, s0, mem0, alive0
    keys = jax.random.split(kw, a.warm + a.recover)
    for t in range(a.warm + a.recover):
        if t == a.warm and spec is not None:
            x, s, mem, alive = damage.apply(spec[0], spec[1], kd, x, s, mem, alive, eps)
        x, s, mem = step(x, s, mem, alive, keys[t])
        if t % a.every == 0 or t == a.warm:
            frames_state[name].append((t, np.asarray(x), np.asarray(npa.colour(s)), np.asarray(alive)))
    print(name, "done", flush=True)

P = a.panel
scale = P / 2.4                       # pixels per world unit: the panel covers [-1.2, 1.2]
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError:
    font = ImageFont.load_default()


def draw(name, rec):
    t, x, col, alive = rec
    im = Image.new("RGB", (P, P), (22, 22, 24))
    dr = ImageDraw.Draw(im)
    r = max(1.6, body_r * scale)
    for i in np.argsort(x[:, 1]):
        if alive[i] <= 0:
            continue
        px, py = P / 2 + x[i, 0] * scale, P / 2 - x[i, 1] * scale
        c = tuple(int(255 * np.clip(v, 0, 1)) for v in col[i])
        dr.ellipse([px - r, py - r, px + r, py + r], fill=c)
    phase = "growing" if t < a.warm else ("DAMAGE" if t == a.warm else "repairing")
    if name == "none" and t >= a.warm:
        phase = "undamaged control"
    dr.text((8, 6), {"none": "no damage", "cut": "cut off", "clump": "squashed", "noise": "scrambled"}[name], fill=(240, 240, 240), font=font)
    dr.text((8, P - 20), f"step {t}  {phase}", fill=(255, 200, 90) if t == a.warm and name != "none" else (170, 170, 170), font=font)
    dr.text((P - 72, 6), f"{int(alive.sum())} bots", fill=(150, 150, 150), font=font)
    return im


frames = []
cnt = len(frames_state["none"])
for i in range(cnt):
    canvas = Image.new("RGB", (2 * P + 6, 2 * P + 6), (0, 0, 0))
    for j, name in enumerate(dmg):
        canvas.paste(draw(name, frames_state[name][i]), ((j % 2) * (P + 6), (j // 2) * (P + 6)))
    frames.append(canvas)
    if frames_state["none"][i][0] == a.warm:
        frames += [canvas] * 5
frames += [frames[-1]] * 8
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
print("saved", a.out, len(frames), "frames", Path(a.out).stat().st_size // 1024, "KB")
