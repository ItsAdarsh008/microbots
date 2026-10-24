"""The baseline MLP cell with a fixed random heading per particle (no shared compass).

Selected with {"kind": "mlph", "heading": "random", ...}. All options of cells.Cell are accepted, plus

    heading       "none" (default: identical to cells.Cell) or "random": particle i has a fixed body-frame angle
                  theta_i = heading.headings(N, heading_seed)[i]. Its network sees the vector features rotated by
                  -theta_i and its proposed move is rotated by +theta_i before the usual squashing.
    heading_seed  seed of the headings (default 0).

A plain MLP is not rotation invariant, so with random headings it has to learn a rule that works in every frame
(or, in effect, cannot use direction at all). Compare with cells_equiv, which is immune by construction.
"""

import jax
import jax.numpy as jnp

import cells
import heading as hd
import npa


class Cell(cells.Cell):
    def __init__(self, **cfg):
        mode = cfg.pop("heading", "none")
        seed = int(cfg.pop("heading_seed", 0))
        if mode not in ("none", "random"):
            raise ValueError(f"heading must be 'none' or 'random', got {mode!r}")
        super().__init__(**cfg)
        self.heading, self.heading_seed = mode, seed
        self.cfg = {**self.cfg, "heading": mode, "heading_seed": seed}

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        z = jnp.concatenate([npa.perceive(x, s, eps * k, eps0 * k, block, total, mass) for k in self.scales], axis=-1)
        rand = self.heading == "random"
        if rand:
            theta = hd.headings(x.shape[0], self.heading_seed)
            z = hd.rotate_perception(z, -theta)
        y = self.apply(p, z, mem)
        dx, ds = y[:, :npa.SPATIAL_DIMS], y[:, npa.SPATIAL_DIMS:npa.OUTPUT_DIMS]
        if rand:
            dx = hd.rotate(dx, theta)
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            m = self.mem_dims
            gate = jax.nn.sigmoid(y[:, npa.OUTPUT_DIMS:npa.OUTPUT_DIMS + m])
            cand = jnp.tanh(y[:, npa.OUTPUT_DIMS + m:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move
