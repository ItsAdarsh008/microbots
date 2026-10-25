"""The random-heading plain cell (cells_mlph) whose wider sensing rings only see a fraction of the bots beyond eps.

Stand-in for line-of-sight occlusion (lab/los_check.py: about a quarter of the second ring is visible in a packed swarm).
A pair (i, j) farther apart than eps is visible with probability ring_vis, decided by a hash of the two positions (so it is
symmetric, fixed while the swarm is still, and changes as bots move); pairs closer than eps are always visible. The first
ring (scale 1) is untouched. Random thinning is an approximation: real occlusion is correlated (bots deep in the swarm see
little at range, bots at the edge see more).

cfg: everything cells_mlph takes, plus ring_vis (0..1, default 0.25).
"""
import jax.numpy as jnp
from jax import lax

import cells_mlph
import heading as hd
import npa


def _vis(x, xi, eps, p):
    """(bs, N) 0/1: pair visible. xi (bs, 2) observers, x (N, 2) everyone. Hash of the pair's summed position."""
    s = xi[:, None, :] + x[None, :, :]
    h = jnp.sin((s[..., 0] * 12.9898 + s[..., 1] * 78.233) * 100.0) * 43758.5453
    u = h - jnp.floor(h)
    d2 = jnp.sum((x[None, :, :] - xi[:, None, :]) ** 2, axis=-1)
    return jnp.where(d2 < eps * eps, 1.0, (u < p).astype(x.dtype))


def perceive_thin(x, s, eps, eps0, near, p, block=None, total=None, mass=None):
    """npa.perceive with the pairs beyond `near` thinned to a fraction p. Same outputs (N, 66)."""
    n = x.shape[0]
    x = lax.stop_gradient(x)
    bs = n if block is None else block
    xb = x.reshape(n // bs, bs, 2)

    def offsets(xi):
        r = x[None, :, :] - xi[:, None, :]
        return r, jnp.sum(r * r, axis=-1)

    m = jnp.ones(n) if mass is None else mass
    rho = lax.map(lambda xi: jnp.sum(npa._poly6(offsets(xi)[1], eps) * _vis(x, xi, near, p) * m[None, :], axis=-1), xb).reshape(n)
    vol = m / jnp.maximum(rho, 1e-12)
    vs = vol[:, None] * s
    sb = s.reshape(n // bs, bs, -1)

    def rows(args):
        xi, si = args
        r, d2 = offsets(xi)
        vis = _vis(x, xi, near, p)
        w = npa._poly6(d2, eps) * vis
        g = npa._spiky_grad(r, d2, eps) * vis[..., None]
        smooth = w @ vs
        grad0 = jnp.einsum("bnd,nc->bcd", g, vs) - si[:, :, None] * jnp.einsum("bnd,n->bd", g, vol)[:, None, :]
        moment = jnp.einsum("n,bna,bnd->bad", vol, r, g)
        inv, det = npa._inv2x2(moment)
        inv = jnp.where((det >= npa.DET_MIN)[:, None, None], inv, jnp.eye(2))
        grad1 = jnp.einsum("bcd,bde->bce", grad0, inv)
        grad = grad0 + lax.stop_gradient(grad1 - grad0)
        drho = jnp.sum(g * m[None, :, None], axis=1)
        return smooth, grad, drho

    smooth, grad, drho = lax.map(rows, (xb, sb))
    smooth = smooth.reshape(n, -1)
    grad = grad.reshape(n, npa.STATE_DIMS, npa.SPATIAL_DIMS)
    drho = drho.reshape(n, npa.SPATIAL_DIMS)
    scale = eps / eps0
    grad = npa._lognorm(grad * scale).reshape(n, -1)
    drho = npa._lognorm(drho * scale ** (1 + npa.SPATIAL_DIMS) / (n if total is None else total))
    return jnp.concatenate([s, smooth, grad, drho], axis=-1)


class Cell(cells_mlph.Cell):
    def __init__(self, **cfg):
        self.ring_vis = float(cfg.pop("ring_vis", 0.25))
        super().__init__(**cfg)
        self.cfg = {**self.cfg, "ring_vis": self.ring_vis}

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        zs = []
        for k in self.scales:
            if k == 1.0:
                zs.append(npa.perceive(x, s, eps * k, eps0 * k, block, total, mass))
            else:
                zs.append(perceive_thin(x, s, eps * k, eps0 * k, eps, self.ring_vis, block, total, mass))
        z = jnp.concatenate(zs, axis=-1)
        if self.heading == "random":
            theta = hd.headings(x.shape[0], self.heading_seed)
            z = hd.rotate_perception(z, -theta)
        y = self.apply(p, z, mem)
        dx, ds = y[:, :npa.SPATIAL_DIMS], y[:, npa.SPATIAL_DIMS:npa.OUTPUT_DIMS]
        if self.heading == "random":
            dx = hd.rotate(dx, theta)
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = npa.ALPHA * dx * eps / (1.0 + length)
        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        if self.mem_dims:
            m = self.mem_dims
            import jax
            gate = jax.nn.sigmoid(y[:, npa.OUTPUT_DIMS:npa.OUTPUT_DIMS + m])
            cand = jnp.tanh(y[:, npa.OUTPUT_DIMS + m:])
            mem = mem + update * (((1 - gate) * mem + gate * cand) - mem)
        return x + update * dx, s + update * ds, mem, move
