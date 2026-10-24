"""An edge-message cell: learned pairwise messages instead of fixed SPH operators.

Selected with {"kind": "edge", ...} in a run config (see cells.build). Same interface as cells.Cell.

For every particle i, its K nearest live neighbours j within eps (neighbors.knn) each send a message

    m_ij = MLP_msg([offset_ij / eps, |offset_ij| / eps, s_i, s_j])

The particle sums the messages of its valid neighbours (divided by K, so the scale does not depend on K) and
also senses how many valid neighbours it has (count / K, a density-like feature). An update MLP reads
[s_i, aggregated message, count feature, mem] (plus the 66-number SPH perception if sph=True) and outputs the
move, the state change and the memory gate/candidate, handled exactly as in cells.Cell.step.

Style follows GNS (https://arxiv.org/abs/2002.09405) and E(n)-GNCA (https://arxiv.org/abs/2301.10497).

    move="direct":  dx comes from the update MLP.
    move="radial":  dx = sum_j valid_ij * offset_ij/eps * phi_x(m_ij) / K  (E(n)-style, equivariant to rotation
                    and reflection when sph=False); the update MLP then outputs only the state/memory parts.
    In both cases dx is squashed as ALPHA * dx * eps / (1 + |dx|).

Only neighbours within eps and among the K nearest are seen: with K below the neighbour count inside eps a
cell sees just its nearest K. Positions are not differentiated through. Translating the swarm changes nothing
the cell senses (only offsets are used), so ds is translation invariant by construction.
"""

import math

import jax
import jax.numpy as jnp

import neighbors
import npa

DEFAULTS = {
    "K": 32,               # neighbours per particle (nearest, within eps)
    "msg_hidden": 32,      # hidden size of the message MLP
    "msg_dims": 32,        # size of a message
    "width": 64,           # hidden size of the update MLP
    "move": "direct",      # direct | radial
    "sph": False,          # also feed the 66-number SPH perception to the update MLP (hybrid)
    "mem": 0,              # private memory channels per particle
    "norm": "none",        # none | rms: RMSNorm on the update MLP's input and hidden layer
    "gain": 0.1,           # scale of the update MLP's output layer init (and of phi_x in radial mode)
}

PAIR_FEATURES = 3          # offset/eps (2) and |offset|/eps (1)


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
        if c["move"] not in ("direct", "radial") or c["norm"] not in ("none", "rms") or int(c["K"]) < 1:
            raise ValueError(f"bad cell config {c}")
        self.K = int(c["K"])
        self.mem_dims = int(c["mem"])
        self.radial = c["move"] == "radial"
        self.sph = bool(c["sph"])
        self.msg_in = PAIR_FEATURES + 2 * npa.STATE_DIMS
        self.in_dims = (npa.STATE_DIMS + int(c["msg_dims"]) + 1 + self.mem_dims
                        + (npa.PERCEPTION_DIMS if self.sph else 0))
        # radial: no move outputs (they come from phi_x), only state change + memory
        self.out_dims = (npa.STATE_DIMS if self.radial else npa.OUTPUT_DIMS) + 2 * self.mem_dims

    # ---- parameters -----------------------------------------------------------
    def init(self, key):
        c = self.cfg
        mh, md, w = int(c["msg_hidden"]), int(c["msg_dims"]), int(c["width"])
        k = iter(jax.random.split(key, 8))
        bias_in = 1.0 / math.sqrt(self.msg_in)
        p = {"m1": _xavier(next(k), self.msg_in, mh, math.sqrt(2.0)),
             "mb1": jax.random.uniform(next(k), (mh,), minval=-bias_in, maxval=bias_in),
             "m2": _xavier(next(k), mh, md, 1.0),
             "mb2": jnp.zeros((md,)),
             "w1": _xavier(next(k), self.in_dims, w, math.sqrt(2.0)),
             "b1": jnp.zeros((w,)),
             "w2": _xavier(next(k), w, self.out_dims, c["gain"])}
        if self.radial:
            p["phi"] = _xavier(next(k), md, 1, c["gain"])      # message -> scalar weight of the offset
        return p

    def param_count(self, params):
        return int(sum(v.size for v in params.values()))

    # ---- forward --------------------------------------------------------------
    def messages(self, p, x, s, mass, eps):
        """Per-pair messages. Returns idx (N, K), valid (N, K), offset/eps (N, K, 2), m (N, K, msg_dims)."""
        n = x.shape[0]
        K = min(self.K, n)
        idx, valid, off = neighbors.knn(x, mass, eps, K)
        off = off / eps
        dist = jnp.sqrt(jnp.sum(off * off, axis=-1, keepdims=True) + 1e-12)
        # first layer on concat([off, dist, s_i, s_j]) == off/dist part + s_i part + s_j part;
        # the s parts are projected once per particle and then gathered, which is cheaper than per pair
        w1 = p["m1"]
        pair = jnp.concatenate([off, dist], axis=-1) @ w1[:PAIR_FEATURES]               # (N, K, mh)
        si = s @ w1[PAIR_FEATURES:PAIR_FEATURES + npa.STATE_DIMS]                       # (N, mh)
        sj = s @ w1[PAIR_FEATURES + npa.STATE_DIMS:]                                    # (N, mh)
        h = jax.nn.relu(pair + si[:, None, :] + sj[idx] + p["mb1"])
        m = h @ p["m2"] + p["mb2"]                                                      # (N, K, md)
        return idx, valid, off, m, K

    def apply(self, p, x, s, mem, eps, eps0, block, total, mass):
        """-> (dx or None, raw update-MLP outputs excluding any move part)."""
        idx, valid, off, m, K = self.messages(p, x, s, mass, eps)
        agg = jnp.sum(valid[..., None] * m, axis=1) / K                                 # (N, md)
        count = (jnp.sum(valid, axis=1, keepdims=True) / K)                             # (N, 1)
        parts = [s, agg, count, mem]
        if self.sph:
            parts.append(npa.perceive(x, s, eps, eps0, block, total, mass))
        z = jnp.concatenate(parts, axis=-1)
        norm = _rms if self.cfg["norm"] == "rms" else (lambda v: v)
        h = jax.nn.relu(norm(z) @ p["w1"] + p["b1"])
        y = norm(h) @ p["w2"]
        dx = None
        if self.radial:
            wgt = (m @ p["phi"])[..., 0]                                                # (N, K)
            dx = jnp.sum((valid * wgt)[..., None] * off, axis=1) / K                    # (N, 2)
        return dx, y

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        """One update of every particle. Returns x, s, mem, mean move length.

        update: (N, 1) 0/1, which particles update. mass: (N,) 1 for live particles, 0 for deleted ones.
        total: only used with sph=True (the particle count the SPH density gradient is divided by).
        """
        n = x.shape[0]
        mass = jnp.ones(n, x.dtype) if mass is None else mass
        dx, y = self.apply(p, x, s, mem, eps, eps0, block, total, mass)
        if self.radial:
            ds, rest = y[:, :npa.STATE_DIMS], y[:, npa.STATE_DIMS:]
        else:
            dx, ds, rest = y[:, :npa.SPATIAL_DIMS], y[:, npa.SPATIAL_DIMS:npa.OUTPUT_DIMS], y[:, npa.OUTPUT_DIMS:]
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            m = self.mem_dims
            gate = jax.nn.sigmoid(rest[:, :m])
            cand = jnp.tanh(rest[:, m:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move
