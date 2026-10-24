"""The growing target (the lizard emoji) and the loss that scores how well a swarm matches it.

The target is a coloured point cloud. To compare it with the swarm, both are "splatted"
onto a picture: every particle becomes a small Gaussian blob of density and colour, so
the loss is smooth in the particle positions and can be trained with gradients. The loss
(paper, section 4.2 and Table 3) compares the density picture and the colour picture.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import npa

HERE = Path(__file__).resolve().parent
LIZARD_PNG = HERE / "data" / "lizard_u1f98e.png"   # Noto Emoji, Apache-2.0

# --- Loss settings: configs/growing.yaml in the paper's release -----------------
ALPHA_MIN = 0.05            # pixels more transparent than this are not part of the shape
TARGET_POINTS = 4096        # about how many points the target is sampled into
AABB = 1.0                  # the target lives in [-1, 1] x [-1, 1]
GRID = 256                  # side of the picture the loss compares
SPLAT_SIGMA = 1.0           # blob width, in target-pixel units
SPLAT_LOSS_WEIGHT = 2.0
DENSITY_WEIGHT = 1.0
COLOUR_WEIGHT = 5.0
DISPLACEMENT_WEIGHT = 0.01  # penalty on how far particles move
OVERFLOW_WEIGHT = 100.0     # penalty on state values outside [-1, 1]
BOUND_WEIGHT = 100.0        # penalty on positions outside [-1, 1]


def load_target(path=LIZARD_PNG, num_points=TARGET_POINTS):
    """Sample the emoji into about `num_points` coloured points. Returns a dict of NumPy arrays."""
    src = Image.open(path).convert("RGBA")
    side = 128
    for _ in range(5):   # pick the picture size that gives about num_points opaque pixels
        img = src.copy()
        img.thumbnail((side, side), Image.LANCZOS)
        inside = (np.asarray(img, dtype=np.float32)[..., 3] / 255.0 >= ALPHA_MIN).sum()
        side = int(round(math.sqrt(num_points / inside) * side))
    img = src.copy()
    img.thumbnail((side, side), Image.LANCZOS)
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = rgba.shape
    pixel = 2.0 * AABB / side
    rows, cols = np.nonzero(rgba[..., 3] >= ALPHA_MIN)
    pos = np.stack([-AABB + (cols + 0.5) * pixel, AABB - (rows + 0.5) * pixel], axis=-1)   # y points up
    return {"pos": pos.astype(np.float32), "rgb": rgba[rows, cols, :3], "pixel": pixel, "side": side}


# =============================================================================
# Splatting
# =============================================================================
def splat(pos, rgb, pixel, grid=GRID, sigma=SPLAT_SIGMA, total=None, weight=None):
    """Draw particles as Gaussian blobs. Returns a (grid, grid, 4) picture: r, g, b, density.

    pos: (N, 2) in [-1, 1], rgb: (N, 3). Each blob adds `rgb * weight` and `weight` to its
    pixels, where the weights are spread so that one blob holds (grid * pixel / 2)^2 of
    density. If `total` is given, the picture is rescaled as if the cloud had that many points.
    """
    unit = grid * pixel / (2.0 * AABB)            # target pixels -> picture pixels
    sig = sigma * unit
    p = (pos + AABB) / (2.0 * AABB) * (grid - 1)
    px, py = p[:, 0], (grid - 1) - p[:, 1]        # image rows run downward
    ax = jnp.arange(grid, dtype=jnp.float32)
    norm = sig * math.sqrt(2.0 * math.pi)         # keeps every blob the same total weight
    wx = jnp.exp(-((ax[None, :] - px[:, None]) ** 2) / (2 * sig * sig)) / norm * unit   # (N, grid)
    wy = jnp.exp(-((ax[None, :] - py[:, None]) ** 2) / (2 * sig * sig)) / norm * unit
    chans = jnp.concatenate([rgb, jnp.ones((rgb.shape[0], 1))], axis=-1)               # (N, 4)
    if weight is not None:                                                             # 0 for deleted particles
        chans = chans * weight[:, None]
    img = jnp.einsum("ny,nx,nc->yxc", wy, wx, chans)
    if total is not None:
        count = pos.shape[0] if weight is None else jnp.maximum(jnp.sum(weight), 1.0)
        img = img * (total / count)
    return img


def make_loss(target, grid=GRID):
    """Build loss(x, s, sum_move) for one batch: x (B, N, 2), s (B, N, 16). Returns (loss, parts)."""
    t_pos = jnp.asarray(target["pos"])
    t_img = splat(t_pos, jnp.asarray(target["rgb"]), target["pixel"], grid)
    t_rgb, t_den = t_img[..., :3], t_img[..., 3:]
    centre = t_pos.mean(axis=0)
    count = t_pos.shape[0]
    pixel = target["pixel"]

    def l1l2(v):
        return jnp.abs(v) + v * v

    def render(x, s, w):
        if w is None:
            x_c = x - x.mean(axis=0) + centre     # the loss does not care where the swarm sits
        else:
            x_c = x - jnp.sum(x * w[:, None], axis=0) / jnp.maximum(jnp.sum(w), 1.0) + centre
        return splat(x_c, npa.colour(s), pixel, grid, total=count, weight=w)

    def loss(x, s, sum_move, alive=None):
        if alive is None:
            img = jax.vmap(lambda x, s: render(x, s, None))(x, s)  # (B, grid, grid, 4)
        else:
            img = jax.vmap(render)(x, s, alive)
        den_diff = l1l2(img[..., 3:] - t_den)
        # the colour only has to match where the density already does (this weight is not trained)
        w = jnp.exp(-jax.lax.stop_gradient(den_diff))
        col_loss = jnp.mean(l1l2(img[..., :3] - t_rgb) * w)
        den_loss = jnp.mean(den_diff)
        splat_loss = DENSITY_WEIGHT * den_loss + COLOUR_WEIGHT * col_loss
        overflow = jnp.mean(jnp.abs(s - jnp.clip(s, -1.0, 1.0)))
        bound = jnp.mean(jnp.abs(x - jnp.clip(x, -1.0, 1.0)))
        total = (SPLAT_LOSS_WEIGHT * splat_loss + DISPLACEMENT_WEIGHT * sum_move
                 + OVERFLOW_WEIGHT * overflow + BOUND_WEIGHT * bound)
        return total, {"density": den_loss, "colour": col_loss, "splat": splat_loss,
                       "move": sum_move, "overflow": overflow, "bound": bound}

    return loss


def l1l2(v):
    return jnp.abs(v) + v * v


def make_eval(target, grid=GRID):
    """Build evaluate(x, s, alive=None) for one swarm: how close is it to the target? Returns a dict.

    density / colour: the two loss terms (lower is better); loss = density + 5 * colour, the training loss
    without its regularisers.
    iou: overlap of the swarm's footprint with the target's, where a pixel counts as
         'filled' if its splatted density is at least half the target's typical density.
    alive: optional (N,) 0/1 mask; deleted particles are ignored and the density is rescaled to the
         surviving count, so the score judges the shape the survivors make, not how many there are.
    """
    t_pos = jnp.asarray(target["pos"])
    t_img = splat(t_pos, jnp.asarray(target["rgb"]), target["pixel"], grid)
    t_rgb, t_den = t_img[..., :3], t_img[..., 3:]
    centre = t_pos.mean(axis=0)
    count, pixel = t_pos.shape[0], target["pixel"]
    t_fill = t_img[..., 3] > 0.5

    def evaluate(x, s, alive=None):
        if alive is None:
            mean = x.mean(axis=0)
        else:
            mean = jnp.sum(x * alive[:, None], axis=0) / jnp.maximum(jnp.sum(alive), 1.0)
        img = splat(x - mean + centre, npa.colour(s), pixel, grid, total=count, weight=alive)
        den_diff = l1l2(img[..., 3:] - t_den)
        colour = jnp.mean(l1l2(img[..., :3] - t_rgb) * jnp.exp(-den_diff))
        density = jnp.mean(den_diff)
        fill = img[..., 3] > 0.5
        iou = jnp.sum(fill & t_fill) / jnp.maximum(jnp.sum(fill | t_fill), 1)
        return {"density": density, "colour": colour, "loss": DENSITY_WEIGHT * density + COLOUR_WEIGHT * colour,
                "iou": iou}

    return evaluate
