"""A rotation-equivariant (SO(2)) cell: works with no shared compass, by construction.

Selected with {"kind": "equiv", ...}. Same interface as cells.Cell.

From npa.perceive it takes the scalars S = [s, smoothed s] (32) and the 17 vectors V (the 16 state-gradient
pairs and the density-gradient pair). A learned linear mix  V'_k = sum_c W[k, c] V_c  (K vectors, shared by all
particles, equivariant because it acts on the channel index, not the coordinates) feeds the invariants

    |V'_k|  (K),   V'_k . V'_l  and  V'_k x V'_l  (K (K-1) / 2 each, k < l).

An MLP reads [S, invariants, mem] and outputs the state change ds (16), per-vector coefficients a_k, b_k (K each)
and, with memory, a gate and a candidate per memory channel. The move is

    dx = sum_k a_k V'_k + b_k J V'_k      (J = rotation by 90 degrees),

squashed as in cells.Cell: ALPHA * dx * eps / (1 + |dx|). Rotating the whole swarm leaves ds and mem unchanged
and rotates dx. Reflections are NOT symmetries (the cross products flip sign), only rotations are.

Options: K (vectors, default 6), width (128), depth (2: one hidden layer like the baseline; more adds ReLU layers),
mem (0), gain (0.1 for the first and last layer), heading ('none' | 'random') and heading_seed (0), the latter
two for tests: random headings (rotate perception by -theta, move by +theta) change nothing.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

import heading as hd
import npa

DEFAULTS = {
    "K": 6,
    "width": 128,
    "depth": 2,
    "mem": 0,
    "gain": 0.1,
    "heading": "none",
    "heading_seed": 0,
}

N_VEC = npa.STATE_DIMS + 1                   # 16 gradient pairs + the density gradient
N_SCALAR = 2 * npa.STATE_DIMS                # s and smoothed s


def _xavier(key, fan_in, fan_out, gain):
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-bound, maxval=bound)


class Cell:
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown cell options {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        if c["heading"] not in ("none", "random") or int(c["K"]) < 1 or int(c["depth"]) < 2:
            raise ValueError(f"bad cell config {c}")
        self.K, self.mem_dims = int(c["K"]), int(c["mem"])
        self.heading, self.heading_seed = c["heading"], int(c["heading_seed"])
        self.n_inv = self.K + self.K * (self.K - 1)       # K norms + K(K-1)/2 dots + K(K-1)/2 crosses
        self.in_dims = N_SCALAR + self.n_inv + self.mem_dims
        self.out_dims = npa.STATE_DIMS + 2 * self.K + 2 * self.mem_dims
        self._pairs = np.triu_indices(self.K, 1)          # (k, l) with k < l

    # ---- parameters -----------------------------------------------------------
    def init(self, key):
        c = self.cfg
        w = c["width"]
        keys = iter(jax.random.split(key, 4 + 2 * c["depth"]))
        bias = 1.0 / math.sqrt(self.in_dims)
        p = {"wv": jax.random.normal(next(keys), (self.K, N_VEC)) / math.sqrt(N_VEC),
             "w1": _xavier(next(keys), self.in_dims, w, c["gain"]),
             "b1": jax.random.uniform(next(keys), (w,), minval=-bias, maxval=bias)}
        for l in range(c["depth"] - 2):
            p[f"wh{l}"] = _xavier(next(keys), w, w, math.sqrt(2.0))
            p[f"bh{l}"] = jnp.zeros((w,))
        p["w2"] = _xavier(next(keys), w, self.out_dims, c["gain"])
        return p

    def param_count(self, params):
        return int(sum(v.size for v in params.values()))

    # ---- forward --------------------------------------------------------------
    def features(self, p, z):
        """z (N, 66) -> scalars (N, 32), mixed vectors V' (N, K, 2), invariants (N, n_inv)."""
        n = z.shape[0]
        scal = z[:, :N_SCALAR]
        grad = z[:, hd.GRAD_START:hd.GRAD_END].reshape(n, npa.STATE_DIMS, 2)
        vec = jnp.concatenate([grad, z[:, hd.GRAD_END:].reshape(n, 1, 2)], axis=1)      # (N, 17, 2)
        v = jnp.einsum("kc,ncd->nkd", p["wv"], vec)                                     # (N, K, 2)
        norm = jnp.sqrt(jnp.sum(v * v, axis=-1) + 1e-12)                                # (N, K)
        i, j = self._pairs
        dot = jnp.sum(v[:, i, :] * v[:, j, :], axis=-1)                                 # (N, K(K-1)/2)
        cross = v[:, i, 0] * v[:, j, 1] - v[:, i, 1] * v[:, j, 0]
        return scal, v, jnp.concatenate([norm, dot, cross], axis=-1)

    def apply(self, p, z, mem):
        """(N, 66) sensed numbers and (N, M) memory -> ds (N, 16), dx (N, 2) unsquashed, memory part (N, 2M)."""
        K = self.K
        scal, v, inv = self.features(p, z)
        h = jax.nn.relu(jnp.concatenate([scal, inv, mem], axis=-1) @ p["w1"] + p["b1"])
        for l in range(self.cfg["depth"] - 2):
            h = jax.nn.relu(h @ p[f"wh{l}"] + p[f"bh{l}"])
        y = h @ p["w2"]
        ds = y[:, :npa.STATE_DIMS]
        a = y[:, npa.STATE_DIMS:npa.STATE_DIMS + K]
        b = y[:, npa.STATE_DIMS + K:npa.STATE_DIMS + 2 * K]
        jv = jnp.stack([-v[..., 1], v[..., 0]], axis=-1)                                # J v
        dx = jnp.einsum("nk,nkd->nd", a, v) + jnp.einsum("nk,nkd->nd", b, jv)
        return ds, dx, y[:, npa.STATE_DIMS + 2 * K:]

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        """One update of every particle. Returns x, s, mem, mean move length (as cells.Cell.step)."""
        z = npa.perceive(x, s, eps, eps0, block, total, mass)
        rand = self.heading == "random"
        if rand:
            theta = hd.headings(x.shape[0], self.heading_seed)
            z = hd.rotate_perception(z, -theta)
        ds, dx, rest = self.apply(p, z, mem)
        if rand:
            dx = hd.rotate(dx, theta)
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            m = self.mem_dims
            gate = jax.nn.sigmoid(rest[:, :m])
            cand = jnp.tanh(rest[:, m:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move
