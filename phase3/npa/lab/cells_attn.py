"""A cell that attends over its neighbours: local multi-head vector attention (Point Transformer style).

Select it with {"kind": "attn", ...} in the cell config (see cells.build).

For particle i, with neighbours j found by neighbors.knn (the K nearest live particles within eps) plus
the particle itself as an always-valid extra slot:

    pe_ij  = MLP_pos(offset_ij / eps)                      offset is 0 for the self slot
    q_i    = W_q [s_i, mem_i]        k_j = W_k s_j        v_j = W_v s_j
    logit_ij = MLP_attn(q_i - k_j + pe_ij)                 one logit per head
    a_ij   = softmax over the valid slots of logit_ij      (invalid slots get -1e9)
    m_i    = sum_j a_ij * (v_j + pe_ij)                    each head owns dim/heads of the channels

The update MLP then reads [m_i, s_i, mem_i] (and, with sph=True, the 66 SPH numbers from npa.perceive)
and produces move, state change and memory gate/candidate, handled exactly as in cells.Cell.step.
Everything is local: a particle only ever reads neighbours within eps. Positions are not differentiated.

    cell = Cell(K=32, heads=4, dim=32, width=128, sph=True, norm="none", mem=0)

Limitation: with K below the number of neighbours in range (about 120 in the lizard), a particle
attends to its nearest K only.
"""

import math

import jax
import jax.numpy as jnp

import npa
from neighbors import knn

DEFAULTS = {
    "K": 32,             # neighbours per particle (nearest first); the self slot comes on top of these
    "heads": 4,          # attention heads; each head has its own logit and dim/heads value channels
    "dim": 32,           # attention feature size (must be a multiple of heads)
    "width": 128,        # hidden size of the update MLP
    "sph": True,         # also feed the update MLP the 66 SPH perception numbers
    "norm": "none",      # none | rms: RMSNorm of the update MLP's whole input vector
    "mem": 0,            # private memory channels per particle (not sensed by neighbours)
    "gain": 0.1,         # scale of the update MLP's first and last layer init (as in the paper)
    "attn_gain": 0.05,   # scale of the attention-logit output layer init (small: weights start near uniform)
    "zero_head": False,  # start with the update MLP's output layer at zero
}

NEG = -1e9


def _xavier(key, fan_in, fan_out, gain=1.0):
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
        if c["norm"] not in ("none", "rms") or c["dim"] % c["heads"] or c["K"] < 1:
            raise ValueError(f"bad cell config {c}")
        self.mem_dims = int(c["mem"])
        self.attn_in = npa.STATE_DIMS + self.mem_dims                  # what the query sees
        self.in_dims = c["dim"] + npa.STATE_DIMS + self.mem_dims + (npa.PERCEPTION_DIMS if c["sph"] else 0)
        self.out_dims = npa.OUTPUT_DIMS + 2 * self.mem_dims

    # ---- parameters -----------------------------------------------------------
    def init(self, key):
        c = self.cfg
        d, h, w = c["dim"], c["heads"], c["width"]
        k = iter(jax.random.split(key, 12))
        bias = 1.0 / math.sqrt(self.in_dims)
        p = {"pos1": _xavier(next(k), 2, d), "pos1b": jnp.zeros((d,)),
             "pos2": _xavier(next(k), d, d), "pos2b": jnp.zeros((d,)),
             "wq": _xavier(next(k), self.attn_in, d),
             "wk": _xavier(next(k), npa.STATE_DIMS, d),
             "wv": _xavier(next(k), npa.STATE_DIMS, d),
             "att1": _xavier(next(k), d, d), "att1b": jnp.zeros((d,)),
             "att2": _xavier(next(k), d, h, c["attn_gain"]), "att2b": jnp.zeros((h,)),
             "w1": _xavier(next(k), self.in_dims, w, c["gain"]),
             "b1": jax.random.uniform(next(k), (w,), minval=-bias, maxval=bias)}
        head = _xavier(next(k), w, self.out_dims, c["gain"])
        p["w2"] = jnp.zeros_like(head) if c["zero_head"] else head
        return p

    def param_count(self, params):
        return int(sum(v.size for v in params.values()))

    # ---- attention ------------------------------------------------------------
    def attention(self, p, x, s, mem, eps, mass=None):
        """Returns m (N, dim), weights (N, K+1, heads), valid (N, K+1), idx (N, K+1). Slot 0 is the particle itself."""
        c = self.cfg
        n, d, h = x.shape[0], c["dim"], c["heads"]
        mass = jnp.ones(n) if mass is None else mass
        idx, valid, off = knn(x, mass, eps, min(c["K"], n))
        idx = jnp.concatenate([jnp.arange(n)[:, None], idx], axis=1)               # (N, K+1)
        valid = jnp.concatenate([jnp.ones((n, 1), valid.dtype), valid], axis=1)
        off = jnp.concatenate([jnp.zeros((n, 1, 2), off.dtype), off], axis=1) / eps

        pe = jax.nn.relu(off @ p["pos1"] + p["pos1b"]) @ p["pos2"] + p["pos2b"]    # (N, K+1, dim)
        q = jnp.concatenate([s, mem], axis=-1) @ p["wq"]                           # (N, dim)
        k = (s @ p["wk"])[idx]                                                     # (N, K+1, dim)
        v = (s @ p["wv"])[idx] + pe
        u = jax.nn.relu((q[:, None, :] - k + pe) @ p["att1"] + p["att1b"]) @ p["att2"] + p["att2b"]   # (N, K+1, heads)
        weights = jax.nn.softmax(jnp.where(valid[..., None] > 0, u, NEG), axis=1)
        m = jnp.einsum("nkh,nkhc->nhc", weights, v.reshape(n, -1, h, d // h)).reshape(n, d)
        return m, weights, valid, idx

    # ---- forward --------------------------------------------------------------
    def apply(self, p, z, m, s, mem):
        parts = [m, s, mem] + ([z] if self.cfg["sph"] else [])
        inp = jnp.concatenate(parts, axis=-1)
        if self.cfg["norm"] == "rms":
            inp = _rms(inp)
        return jax.nn.relu(inp @ p["w1"] + p["b1"]) @ p["w2"]

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        """One update of every particle. Returns x, s, mem, mean move length (same contract as cells.Cell.step)."""
        m = self.attention(p, x, s, mem, eps, mass)[0]
        z = npa.perceive(x, s, eps, eps0, block, total, mass) if self.cfg["sph"] else None
        y = self.apply(p, z, m, s, mem)
        dx, ds = y[:, :npa.SPATIAL_DIMS], y[:, npa.SPATIAL_DIMS:npa.OUTPUT_DIMS]
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            mm = self.mem_dims
            gate = jax.nn.sigmoid(y[:, npa.OUTPUT_DIMS:npa.OUTPUT_DIMS + mm])
            cand = jnp.tanh(y[:, npa.OUTPUT_DIMS + mm:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move
