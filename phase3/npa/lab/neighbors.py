"""Fixed-size neighbour lists, for cells that look at individual neighbours instead of SPH sums.

    idx, valid, offset = knn(x, mass, eps, K)

For every particle: the indices of its K nearest live neighbours closer than eps (never itself), a 0/1
mask for slots that hold a real neighbour, and the offsets x_j - x_i. A particle with fewer than K
neighbours gets masked slots. With about 120 neighbours inside eps in the lizard, K < 120 means a cell
sees only its nearest K, not every neighbour in range; say so when reporting results.

Positions are not differentiated through (the paper stops gradients through positions too).
"""

import jax
import jax.numpy as jnp
from jax import lax

FAR = 1e9


def knn(x, mass, eps, K):
    x = lax.stop_gradient(x)
    n = x.shape[0]
    d2 = jnp.sum((x[None, :, :] - x[:, None, :]) ** 2, axis=-1)           # d2[i, j]
    ok = (d2 < eps * eps) & (mass[None, :] > 0) & ~jnp.eye(n, dtype=bool)
    nearest, idx = lax.top_k(-jnp.where(ok, d2, FAR), K)
    valid = (-nearest < FAR / 2).astype(x.dtype)
    return idx, valid, x[idx] - x[:, None, :]
