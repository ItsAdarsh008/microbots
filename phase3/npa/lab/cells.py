"""The per-particle update networks ("cells"), all behind one interface so experiments can swap them.

A cell takes what a particle senses (the 66 numbers from npa.perceive, plus the particle's own
private memory, if the cell has any) and returns a move, a change to its 16 state channels, and its
new memory. Everything outside the cell (neighbour sensing, the 50% update mask, damage, the loss)
is shared, so a difference in results is a difference in the cell.

    cell = Cell(block="gated", width=128, depth=3, norm="rms", mem=8)
    params = cell.init(key)
    x, s, mem, move = cell.step(params, x, s, mem, update, eps, eps0)

Cell("plain", width=128, depth=2) is exactly the paper's network (same parameter names as npa.py).
"""

import math

import jax
import jax.numpy as jnp

import npa

DEFAULTS = {
    "block": "plain",       # plain: relu layers | residual: h + relu-MLP(h) | gated: h + SwiGLU(h)
    "width": 128,           # hidden size
    "depth": 2,             # plain: number of linear layers | residual/gated: depth - 1 blocks
    "ff": None,             # inner size of a residual/gated block (default: width)
    "norm": "none",         # none | rms: RMSNorm in front of each block and the head
    "mem": 0,               # private memory channels per particle (not sensed by neighbours)
    "scales": (1.0,),       # sensing radii as multiples of eps; (1, 2) also senses at 2 eps (still local, a bit wider)
    "gain": 0.1,            # scale of the first and last layer's random init (the paper uses 0.1)
    "zero_head": False,     # start with the output layer at zero, so nothing moves at first
}


def _xavier(key, fan_in, fan_out, gain):
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-bound, maxval=bound)


def _rms(v):
    return v * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + 1e-6)


class Cell:
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown cell options {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        if c["block"] not in ("plain", "residual", "gated") or c["norm"] not in ("none", "rms"):
            raise ValueError(f"bad cell config {c}")
        self.mem_dims = int(c["mem"])
        self.scales = tuple(float(v) for v in c["scales"])
        self.in_dims = npa.PERCEPTION_DIMS * len(self.scales) + self.mem_dims
        self.out_dims = npa.OUTPUT_DIMS + 2 * self.mem_dims    # + a gate and a candidate per memory channel

    # ---- parameters -----------------------------------------------------------
    def init(self, key):
        c = self.cfg
        w, ff = c["width"], c["ff"] or c["width"]
        keys = iter(jax.random.split(key, 4 + 4 * max(c["depth"], 2)))
        bias = 1.0 / math.sqrt(self.in_dims)
        p = {"w1": _xavier(next(keys), self.in_dims, w, c["gain"]),
             "b1": jax.random.uniform(next(keys), (w,), minval=-bias, maxval=bias)}
        if c["block"] == "plain":
            for l in range(c["depth"] - 2):
                p[f"wh{l}"] = _xavier(next(keys), w, w, math.sqrt(2.0))
                p[f"bh{l}"] = jnp.zeros((w,))
        else:
            for l in range(max(c["depth"] - 1, 1)):
                if c["block"] == "residual":
                    p[f"blk{l}_a"], p[f"blk{l}_ab"] = _xavier(next(keys), w, ff, math.sqrt(2.0)), jnp.zeros((ff,))
                    p[f"blk{l}_b"] = _xavier(next(keys), ff, w, 1.0) * 0.5
                else:
                    p[f"blk{l}_g"], p[f"blk{l}_u"] = _xavier(next(keys), w, ff, 1.0), _xavier(next(keys), w, ff, 1.0)
                    p[f"blk{l}_d"] = _xavier(next(keys), ff, w, 1.0) * 0.5
        head = _xavier(next(keys), w, self.out_dims, c["gain"])
        p["w2"] = jnp.zeros_like(head) if c["zero_head"] else head
        return p

    def param_count(self, params):
        return int(sum(v.size for v in params.values()))

    # ---- forward --------------------------------------------------------------
    def apply(self, p, z, mem):
        """(N, 66) sensed numbers and (N, M) memory -> (N, 18 + 2M) raw outputs."""
        c = self.cfg
        norm = _rms if c["norm"] == "rms" else (lambda v: v)
        h = jnp.concatenate([z, mem], axis=-1) @ p["w1"] + p["b1"]
        if c["block"] == "plain":
            h = jax.nn.relu(h)
            for l in range(c["depth"] - 2):
                h = jax.nn.relu(h @ p[f"wh{l}"] + p[f"bh{l}"])
        else:
            for l in range(max(c["depth"] - 1, 1)):
                n = norm(h)
                if c["block"] == "residual":
                    h = h + jax.nn.relu(n @ p[f"blk{l}_a"] + p[f"blk{l}_ab"]) @ p[f"blk{l}_b"]
                else:
                    h = h + (jax.nn.silu(n @ p[f"blk{l}_g"]) * (n @ p[f"blk{l}_u"])) @ p[f"blk{l}_d"]
            h = norm(h)
        return h @ p["w2"]

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        """One update of every particle. Returns x, s, mem, mean move length.

        update: (N, 1) 0/1, which particles update. mass: (N,) 1 for live particles, 0 for deleted ones.
        total: the particle count the density gradient is normalised by (the original swarm size).
        """
        z = jnp.concatenate([npa.perceive(x, s, eps * k, eps0 * k, block, total, mass) for k in self.scales], axis=-1)
        y = self.apply(p, z, mem)
        dx, ds = y[:, :npa.SPATIAL_DIMS], y[:, npa.SPATIAL_DIMS:npa.OUTPUT_DIMS]
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            m = self.mem_dims
            gate = jax.nn.sigmoid(y[:, npa.OUTPUT_DIMS:npa.OUTPUT_DIMS + m])
            cand = jnp.tanh(y[:, npa.OUTPUT_DIMS + m:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move


def build(**cfg):
    """Make a cell from a config. cfg["kind"] picks the family: "mlp" (this file) or any module
    named cells_<kind>.py in lab/, which must define a Cell class with the same interface.
    cfg["embody"] = {...} wraps the cell in a physical-robot layer (lab/embody.py): limited speed, noisy
    quantised sensing, packet loss, no overlap, or hops on a docking lattice. See embodiment.md."""
    embody_spec = cfg.pop("embody", None)
    kind = cfg.pop("kind", "mlp")
    if kind == "mlp":
        cell = Cell(**cfg)
    else:
        import importlib
        cell = importlib.import_module(f"cells_{kind}").Cell(**cfg)
    if embody_spec:
        import embody
        cell = embody.Embodied(cell, **embody_spec)
    return cell
