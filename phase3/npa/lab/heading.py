"""Per-particle fixed headings: the real bots have no shared compass.

Each bot i measures neighbour bearings in its own body frame, rotated by an unknown fixed angle theta_i from the
world frame. The simulation emulates this exactly: the global-frame vector features of npa.perceive are
rotated by -theta_i before the bot's network sees them, and the move the network proposes (in its body frame)
is rotated by +theta_i before it is executed in the world.

    theta = heading.headings(n, seed)                      # (n,) angles in [0, 2 pi)
    heading.rotate(v, theta)                               # R(theta) v for (..., 2) vectors
    heading.rotate_perception(z, theta)                    # rotate only the vector blocks of a perceive() output
"""

import math

import jax
import jax.numpy as jnp

import npa

PERCEPTION = npa.PERCEPTION_DIMS      # 66 = s (16) | smoothed s (16) | grad (16 x 2) | density grad (2)
GRAD_START = 2 * npa.STATE_DIMS       # 32
GRAD_END = GRAD_START + npa.STATE_DIMS * npa.SPATIAL_DIMS   # 64


def headings(n, seed=0):
    """(n,) angles uniform in [0, 2 pi), deterministic in (n, seed)."""
    return jax.random.uniform(jax.random.PRNGKey(seed), (n,), minval=0.0, maxval=2.0 * math.pi)


def rotate(v, theta):
    """R(theta) v for vectors v (..., 2); theta broadcasts against v[..., 0] (e.g. (n,) for (n, 2))."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def rotate_perception(z, theta):
    """Rotate the vector parts of a perceive output by per-particle angles theta (n,). z: (n, 66*k), where k
    is the number of concatenated sensing scales. Scalars (state, smoothed state) are left untouched."""
    n, d = z.shape
    assert d % PERCEPTION == 0, "expected a multiple of 66 numbers per particle"
    k = d // PERCEPTION
    zz = z.reshape(n, k, PERCEPTION)
    th = theta[:, None, None]                                  # broadcast over scales and channels
    grad = rotate(zz[..., GRAD_START:GRAD_END].reshape(n, k, npa.STATE_DIMS, 2), th)
    drho = rotate(zz[..., GRAD_END:], theta[:, None])
    zz = jnp.concatenate([zz[..., :GRAD_START], grad.reshape(n, k, -1), drho], axis=-1)
    return zz.reshape(n, d)
