"""Task 2b (shape-agnostic healing): the shapes, the swarms sampled from them, and the score.

Nothing about a shape is stored in the weights. A swarm is sampled from a coloured mask: the particle POSITIONS
are the shape and the COLOUR channels (last three state channels, colour = s[-3:] + 0.5) hold its pattern.
The reference picture is the swarm at its start, rendered with target.splat, so any shape works.

Shape sources
    emoji        TRAIN_EMOJI (training) and HELDOUT_EMOJI (evaluation), loaded from lab/shapes.py's PNGs
    procedural   proc_shape(seed): 2-5 discs / ellipses / thick segments (branches), 0-3 holes, a smooth random colour
                 field with optional stripes. Training uses seeds >= 1000, the held-out suite uses seeds 0..19.

Every shape is a (128, 128) bool mask and a (128, 128, 3) float rgb picture covering [-1, 1]^2 (row 0 is y = +1).

Density normalisation. target.splat draws one blob per particle and, given `total`, rescales the picture as if the cloud
had that many points. For the lizard target total = 4096 points at one point per target pixel, so the density inside
the shape is about 1 (the IoU threshold 0.5 relies on this). Here total = mask area / PIXEL^2, the number of target-pixel
sized cells in the shape, so the density inside is about 1 for every shape whatever n is. (With total = n the density
would be about PIXEL^2 / spacing^2, around 0.2, and the IoU threshold would never fire.) `shape_total` computes it and
`score` takes it as an argument (default: the number of particles, which is the literal alternative).
"""

import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import npa  # noqa: E402
import shapes  # noqa: E402
import target as tgt  # noqa: E402

FRAME = 128
PIXEL = 2.0 / 124           # the lizard target's pixel: the same blob width for every shape
ZOOM = 0.8                  # the picture covers [-1/ZOOM, 1/ZOOM]^2 around the swarm's mean (about 29% of shapes reach past 1.0 from it);
                            # the blob width in world units is still PIXEL, only the picture is wider
GRID = 128                  # side of the picture the loss compares (training default; the evaluator may differ)
ALPHA_MIN = 0.4             # an emoji pixel counts as shape above this opacity
MARGIN = 0.92               # a procedural shape's longer side is scaled to span 2 * MARGIN of the frame
MIN_HALF_WIDTH = 0.05       # thinnest branch: half-width 0.05 = a thickness of 0.05 of the 2-wide frame

TRAIN_EMOJI = ("star", "bone", "fish", "cactus", "anchor", "wrench", "ring", "key")
HELDOUT_EMOJI = ("lizard", "butterfly", "octopus", "donut", "snowflake")
HELDOUT_PROC_SEEDS = tuple(range(20))
TRAIN_PROC_MIN_SEED = 1000

# genus of the held-out procedural seeds, as computed by genus(proc_shape(seed)[0]); test_agnostic.py checks the table
PROC_GENUS = {
    0: 0, 1: 3, 2: 1, 3: 1, 4: 0, 5: 0, 6: 2, 7: 0, 8: 2, 9: 0,
    10: 0, 11: 2, 12: 3, 13: 3, 14: 3, 15: 1, 16: 1, 17: 2, 18: 1, 19: 3,
}

_AX = -1.0 + (np.arange(FRAME) + 0.5) * 2.0 / FRAME
PX = np.broadcast_to(_AX[None, :], (FRAME, FRAME)).astype(np.float32)        # world x of every pixel centre
PY = np.broadcast_to(-_AX[:, None], (FRAME, FRAME)).astype(np.float32)       # world y (rows run downward)


# =============================================================================
# Topology
# =============================================================================
def _label(mask, eight):
    """Connected components of a bool image by flood fill. Returns (labels, count); background is 0."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if eight else [])
    count = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        count += 1
        lab[r0, c0] = count
        queue = deque([(r0, c0)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = count
                    queue.append((rr, cc))
    return lab, count


def n_pieces(mask):
    """Number of connected pieces of the shape (8-connected)."""
    return _label(mask, True)[1]


def genus(mask):
    """Number of holes: background components (4-connected, outside border included) minus 1."""
    padded = np.pad(~mask, 1, constant_values=True)
    return _label(padded, False)[1] - 1


# =============================================================================
# Emoji
# =============================================================================
def _inpaint(rgb, mask, iters=12):
    """Fill the colours outside the mask with the average of coloured neighbours, so bilinear lookups at the rim are clean."""
    rgb, known = rgb.copy(), mask.copy()
    for _ in range(iters):
        if known.all():
            break
        acc, cnt = np.zeros_like(rgb), np.zeros(mask.shape, np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                k = np.roll(known, (dr, dc), (0, 1))
                acc += np.roll(rgb * known[..., None], (dr, dc), (0, 1))
                cnt += k
        fill = ~known & (cnt > 0)
        rgb[fill] = acc[fill] / cnt[fill][:, None]
        known = known | fill
    return rgb


def _clean(mask, speck=6):
    """Drop tiny pieces and fill tiny holes left by anti-aliasing."""
    lab, k = _label(mask, True)
    for i in range(1, k + 1):
        if (lab == i).sum() < speck * 2:
            mask = mask & (lab != i)
    padded = np.pad(~mask, 1, constant_values=True)
    lab, k = _label(padded, False)
    for i in range(1, k + 1):
        comp = lab == i
        if comp.sum() < speck and not comp[0].any() and not comp[-1].any() and not comp[:, 0].any() and not comp[:, -1].any():
            padded = padded & ~comp
    return ~padded[1:-1, 1:-1]


@lru_cache(maxsize=None)
def _emoji(name):
    img = Image.open(shapes.path(name)).convert("RGBA")
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    arr = np.asarray(canvas.resize((FRAME, FRAME), Image.LANCZOS), dtype=np.float32) / 255.0
    mask = _clean(arr[..., 3] >= ALPHA_MIN)
    rgb = _inpaint(arr[..., :3], mask)
    mask.flags.writeable = rgb.flags.writeable = False
    return mask, rgb


def emoji_shape(name):
    """(mask (128, 128) bool, rgb (128, 128, 3) float) of an emoji from lab/shapes.py."""
    return _emoji(name)


# =============================================================================
# Procedural shapes
# =============================================================================
def _inside(parts, qx, qy):
    out = np.zeros(qx.shape, bool)
    for p in parts:
        if p[0] == "disc":
            _, cx, cy, r = p
            out |= (qx - cx) ** 2 + (qy - cy) ** 2 < r * r
        elif p[0] == "ell":
            _, cx, cy, a, b, th = p
            u = (qx - cx) * np.cos(th) + (qy - cy) * np.sin(th)
            v = -(qx - cx) * np.sin(th) + (qy - cy) * np.cos(th)
            out |= (u / a) ** 2 + (v / b) ** 2 < 1.0
        else:
            _, x0, y0, x1, y1, hw = p
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((qx - x0) * dx + (qy - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
            out |= (qx - (x0 + t * dx)) ** 2 + (qy - (y0 + t * dy)) ** 2 < hw * hw
    return out


def _random_part(rng, anchor):
    kind = rng.choice(["disc", "ell", "seg", "seg"])
    cx, cy = anchor
    if kind == "disc":
        return ("disc", cx, cy, rng.uniform(0.1, 0.4))
    if kind == "ell":
        a = rng.uniform(0.2, 0.55)
        return ("ell", cx, cy, a, rng.uniform(0.07, min(0.25, a)), rng.uniform(0, np.pi))
    ang, length = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.9)
    hw = MIN_HALF_WIDTH + 0.09 * rng.uniform() ** 2
    return ("seg", cx, cy, cx + length * np.cos(ang), cy + length * np.sin(ang), hw)


def _colour_field(rng, mask):
    """Smooth random colours: a few low-frequency sinusoids per channel, plus (usually) random stripes."""
    rgb = np.zeros((FRAME, FRAME, 3), np.float32)
    for c in range(3):
        field = 0.5 + rng.uniform(-0.12, 0.12)
        for _ in range(3):
            ang, freq = rng.uniform(0, 2 * np.pi), rng.uniform(1.0, 4.5)
            field = field + rng.uniform(0.08, 0.22) * np.sin(freq * (np.cos(ang) * PX + np.sin(ang) * PY) + rng.uniform(0, 2 * np.pi))
        rgb[..., c] = field
    if rng.random() < 0.7:
        ang, freq = rng.uniform(0, 2 * np.pi), rng.uniform(5.0, 14.0)
        stripes = np.tanh(3.0 * np.sin(freq * (np.cos(ang) * PX + np.sin(ang) * PY) + rng.uniform(0, 2 * np.pi)))
        rgb += stripes[..., None] * rng.uniform(-0.3, 0.3, 3).astype(np.float32)
    return np.clip(rgb, 0.03, 0.97).astype(np.float32)


@lru_cache(maxsize=4096)
def proc_shape(seed):
    """A random shape with branches, holes and thin parts. Returns (mask (128, 128) bool, rgb (128, 128, 3) float).

    Deterministic in `seed`. Connected by construction (each part is anchored inside the union so far); holes are
    discs cut out with a wall of at least 0.05 around them, so a hole never breaks the shape or merges with the outside.
    """
    rng = np.random.default_rng([int(seed), 20260101])
    parts = []
    for i in range(int(rng.integers(2, 6))):
        for _ in range(30):
            if parts:
                cur = _inside(parts, PX, PY)
                k = rng.integers(cur.sum())
                anchor = (float(PX[cur][k]), float(PY[cur][k]))
            else:
                anchor = tuple(rng.uniform(-0.3, 0.3, 2))
            part = _random_part(rng, anchor)
            one = _inside([part], PX, PY)
            if one.any() and np.abs(PX[one]).max() < 0.95 and np.abs(PY[one]).max() < 0.95:
                parts.append(part)
                break
    m1 = _inside(parts, PX, PY)
    xs, ys = PX[m1], PY[m1]
    w, h = xs.max() - xs.min() + 2 / FRAME, ys.max() - ys.min() + 2 / FRAME
    sc = float(np.clip(2 * MARGIN / max(w, h), 0.6, 1.8))
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    mask = _inside(parts, cx + PX / sc, cy + PY / sc)      # same parts, evaluated at the zoomed frame

    n_holes = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.3, 0.25, 0.2]))
    for _ in range(n_holes):
        for attempt in range(60):
            k = rng.integers(mask.sum())
            hx, hy = float(PX[mask][k]), float(PY[mask][k])
            r = rng.uniform(0.07, 0.22) * (0.95 ** (attempt // 4))
            near = (PX - hx) ** 2 + (PY - hy) ** 2 < (r + 0.05) ** 2
            if mask[near].all() and near.sum() > 0:
                mask = mask & ~((PX - hx) ** 2 + (PY - hy) ** 2 < r * r)
                break
    rgb = _colour_field(rng, mask)
    mask.flags.writeable = rgb.flags.writeable = False
    return mask, rgb


# =============================================================================
# The suite
# =============================================================================
def get_shape(name):
    """'star' -> an emoji; 'proc:17' -> proc_shape(17). Returns (mask, rgb)."""
    return proc_shape(int(name[5:])) if name.startswith("proc:") else emoji_shape(name)


def heldout_suite(emoji=HELDOUT_EMOJI, proc_seeds=HELDOUT_PROC_SEEDS):
    """The held-out shapes: a list of dicts with name, source ('emoji' or 'proc'), genus, mask, rgb."""
    out = []
    for name in emoji:
        mask, rgb = emoji_shape(name)
        out.append({"name": name, "source": "emoji", "genus": genus(mask), "mask": mask, "rgb": rgb})
    for sd in proc_seeds:
        mask, rgb = proc_shape(int(sd))
        out.append({"name": f"proc:{sd}", "source": "proc", "genus": genus(mask), "mask": mask, "rgb": rgb})
    return out


# =============================================================================
# Swarms
# =============================================================================
def _rng(seed):
    return np.random.default_rng(np.asarray(seed).ravel().astype(np.uint32).tolist() if hasattr(seed, "shape") else int(seed))


def _lookup(mask, x):
    """Is each world point inside the mask (nearest pixel)?"""
    c = np.floor((x[:, 0] + 1.0) / 2.0 * FRAME).astype(int)
    r = np.floor((1.0 - x[:, 1]) / 2.0 * FRAME).astype(int)
    ok = (c >= 0) & (c < FRAME) & (r >= 0) & (r < FRAME)
    return ok & mask[np.clip(r, 0, FRAME - 1), np.clip(c, 0, FRAME - 1)]


def _bilinear(img, x):
    """Sample an (H, W, C) picture covering [-1, 1]^2 at world points x (M, 2)."""
    col = np.clip((x[:, 0] + 1.0) / 2.0 * FRAME - 0.5, 0, FRAME - 1)
    row = np.clip((1.0 - x[:, 1]) / 2.0 * FRAME - 0.5, 0, FRAME - 1)
    c0, r0 = np.minimum(col.astype(int), FRAME - 2), np.minimum(row.astype(int), FRAME - 2)
    fc, fr = (col - c0)[:, None], (row - r0)[:, None]
    return ((img[r0, c0] * (1 - fc) + img[r0, c0 + 1] * fc) * (1 - fr) + (img[r0 + 1, c0] * (1 - fc) + img[r0 + 1, c0 + 1] * fc) * fr)


def _hex(a, theta, offset):
    k = int(np.ceil(1.6 / (a * np.sqrt(3) / 2))) + 2
    i, j = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    p = np.stack([a * (i + 0.5 * j), a * np.sqrt(3) / 2 * j], axis=-1).reshape(-1, 2)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = p @ rot.T + offset
    return p[(np.abs(p) <= 1.0).all(axis=1)]


def _drop_closest(pts, k, rng):
    """Remove k points, always one of the closest pair, so the packing stays even."""
    m = len(pts)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).astype(np.float32)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(1)
    keep = np.ones(m, bool)
    for _ in range(k):
        dist = np.where(keep, d[np.arange(m), nn], np.inf)
        i = int(dist.argmin())
        j = int(nn[i])
        gone = i if rng.random() < 0.5 else j
        keep[gone] = False
        d[gone, :] = np.inf
        d[:, gone] = np.inf
        for r in np.nonzero(keep & (nn == gone))[0]:
            nn[r] = d[r].argmin()
    return pts[keep]


def _add_best(pts, k, mask, rng, cands=10):
    """Add k points inside the mask, each the best of a few random candidates (farthest from the existing points)."""
    rows, cols = np.nonzero(mask)
    cur = pts
    for _ in range(k):
        pick = rng.integers(len(rows), size=cands)
        cand = np.stack([-1 + (cols[pick] + rng.random(cands)) * 2 / FRAME, 1 - (rows[pick] + rng.random(cands)) * 2 / FRAME], -1)
        dmin = np.sqrt(((cand[:, None] - cur[None]) ** 2).sum(-1)).min(1)
        new = cand[dmin.argmax()][None]
        cur = np.concatenate([cur, new])
    return cur


def sample_swarm(mask, rgb, n, seed):
    """n near-uniformly packed particles inside the mask. Returns x (n, 2) in world coordinates (y up) and s (n, 16)
    with s[:, -3:] = rgb at the particle - 0.5 and every other channel 0. Deterministic given `seed` (an int or a PRNG key)."""
    rng = _rng(seed)
    area = mask.sum() * (2.0 / FRAME) ** 2
    a = float(np.sqrt(area / n * 2.0 / np.sqrt(3)))
    theta, offset = rng.uniform(0, np.pi / 3), rng.uniform(-a, a, 2)
    best = None
    for _ in range(8):             # tune the spacing until about n lattice points land inside
        p = _hex(a, theta, offset)
        p = p + rng.uniform(-0.15 * a, 0.15 * a, p.shape)
        p = p[_lookup(mask, p)]
        if best is None or abs(len(p) - n) < abs(len(best) - n):
            best = p
        if abs(len(p) - n) <= max(1, n // 200):
            break
        a *= float(np.sqrt(len(p) / n)) ** 0.9
    p = best
    if len(p) > n:
        p = _drop_closest(p, len(p) - n, rng)
    elif len(p) < n:
        p = _add_best(p, n - len(p), mask, rng)
    p = p[rng.permutation(n)].astype(np.float32)
    s = np.zeros((n, npa.STATE_DIMS), np.float32)
    s[:, -3:] = _bilinear(rgb, p) - 0.5
    return p, s


def shape_total(mask):
    """The `total` for splat: the shape's area in target-pixel sized cells (see the module docstring)."""
    return float(mask.sum() * (2.0 / FRAME) ** 2 / PIXEL**2)


def train_batch(rng, batch, n, proc_fraction=0.7):
    """Fresh training swarms: numpy x (B, n, 2), s (B, n, 16), total (B,). rng: a numpy Generator."""
    xs, ss, tot = [], [], []
    for _ in range(batch):
        if rng.random() < proc_fraction:
            mask, rgb = proc_shape(int(rng.integers(TRAIN_PROC_MIN_SEED, TRAIN_PROC_MIN_SEED + 1_000_000)))
        else:
            mask, rgb = emoji_shape(TRAIN_EMOJI[int(rng.integers(len(TRAIN_EMOJI)))])
        x, s = sample_swarm(mask, rgb, n, int(rng.integers(1 << 30)))
        xs.append(x), ss.append(s), tot.append(shape_total(mask))
    return np.stack(xs), np.stack(ss), np.asarray(tot, np.float32)


# =============================================================================
# Reference image and score (jax, vmappable)
# =============================================================================
def centre(x, alive=None):
    """The swarm's own (weighted) mean position."""
    if alive is None:
        return x.mean(axis=0)
    return jnp.sum(x * alive[:, None], axis=0) / jnp.maximum(jnp.sum(alive), 1.0)


def render(x, s, alive=None, total=None, grid=GRID):
    """(grid, grid, 4) picture r, g, b, density of one swarm, centred on its own mean, with the same blob width for every shape."""
    total = x.shape[0] if total is None else total
    return tgt.splat(ZOOM * (x - centre(x, alive)), npa.colour(s), PIXEL * ZOOM, grid, total=total, weight=alive)


def score(x, s, alive, ref_img, total=None):
    """How close is the swarm to its start? Computed like target.make_eval but against `ref_img`
    (= render of the start swarm): dict(density, colour, loss = density + 5 colour, iou)."""
    img = render(x, s, alive, total, ref_img.shape[0])
    ref_rgb, ref_den = ref_img[..., :3], ref_img[..., 3:]
    den_diff = tgt.l1l2(img[..., 3:] - ref_den)
    colour = jnp.mean(tgt.l1l2(img[..., :3] - ref_rgb) * jnp.exp(-den_diff))
    density = jnp.mean(den_diff)
    fill, ref_fill = img[..., 3] > 0.5, ref_img[..., 3] > 0.5
    iou = jnp.sum(fill & ref_fill) / jnp.maximum(jnp.sum(fill | ref_fill), 1)
    return {"density": density, "colour": colour, "loss": tgt.DENSITY_WEIGHT * density + tgt.COLOUR_WEIGHT * colour, "iou": iou}
