"""Phase 3: a Neural Particle Automaton (NPA) in JAX.

A reimplementation of "Neural Particle Automata: Learning Self-Organizing
Particle Dynamics" (Kim, Pajouheshgar, Susstrunk, Jakob, Park; arXiv 2601.16096).
Written from the paper's equations, not copied from the authors' code.

Every particle carries a position and a 16-number state, and every particle runs
the same tiny neural network. Each step a particle:
  1. looks at the neighbours within EPS of it, using smoothed-particle-hydrodynamics
     (SPH) operators: its own state, the smoothed state around it, how the state
     changes across space (a gradient) and how crowded it is (a density gradient)
  2. feeds those 66 numbers through a 2-layer MLP, which outputs how far to move
     and how to change its state
  3. applies the update, but only with probability 0.5 (so particles are not in sync)

The last three state channels are the particle's colour, so the swarm can paint
a picture while it moves. Nothing in the rule knows the target shape: the shape
comes from training the MLP so that a seed blob grows into it and heals itself.

Neighbours are found by brute force (every particle against every other), written
as matrix products so it runs well on a CPU. Fine for 1,000 to 4,000 particles.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

# --- Model size (paper, Table 3: 2D growing) -----------------------------------
STATE_DIMS = 16
HIDDEN_DIMS = 128
SPATIAL_DIMS = 2
PERCEPTION_DIMS = STATE_DIMS * (2 + SPATIAL_DIMS) + SPATIAL_DIMS   # 66
OUTPUT_DIMS = SPATIAL_DIMS + STATE_DIMS                            # 18

# --- Dynamics ------------------------------------------------------------------
ALPHA = 0.5          # scales how far a particle can move in one step (in units of EPS)
UPDATE_PROB = 0.5    # each particle updates with this probability each step
DET_MIN = 1e-3       # skip the gradient correction where the moment matrix is this flat


# =============================================================================
# 1. Parameters
# =============================================================================
def init_params(key, gain=0.1):
    """Random weights: Xavier-uniform scaled down by `gain`, so the first steps barely move."""
    k1, k2, k3 = jax.random.split(key, 3)

    def xavier(k, fan_in, fan_out):
        bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(k, (fan_in, fan_out), minval=-bound, maxval=bound)

    bias_bound = 1.0 / math.sqrt(PERCEPTION_DIMS)
    return {
        "w1": xavier(k1, PERCEPTION_DIMS, HIDDEN_DIMS),
        "b1": jax.random.uniform(k2, (HIDDEN_DIMS,), minval=-bias_bound, maxval=bias_bound),
        "w2": xavier(k3, HIDDEN_DIMS, OUTPUT_DIMS),   # the output layer has no bias
    }


# =============================================================================
# 2. SPH operators
# =============================================================================
# All kernels are zero beyond distance eps. rij = xj - xi points from i to j.
#   poly6:  smooth bump, used for density and for smoothing a state
#   spiky:  the kernel's gradient, used for gradients (it stays steep near r = 0)
def _poly6(d2, eps):
    norm = 4.0 / (math.pi * eps**8)
    return norm * jnp.maximum(eps * eps - d2, 0.0) ** 3


def _spiky_grad(r, d2, eps):
    """Gradient of the spiky kernel at offset r (pointing toward the neighbour), shape (..., 2)."""
    norm = 10.0 / (math.pi * eps**5)
    d = jnp.sqrt(d2 + 1e-12)
    inside = (d2 < eps * eps) & (d2 > 0.0)    # a particle contributes nothing to itself
    mag = jnp.where(inside, norm * 3.0 * (eps - d) ** 2 / d, 0.0)
    return mag[..., None] * r


def _lognorm(v):
    """Squash the length of each vector with log(1 + |v|), keeping its direction."""
    n = jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-24)
    return jnp.log1p(n) * v / n


def _inv2x2(M):
    """Inverse of a batch of 2x2 matrices, and their determinants."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    safe = jnp.where(jnp.abs(det) > 1e-12, det, 1.0)
    inv = jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / safe[..., None, None]
    return inv, det


def perceive(x, s, eps, eps0, block=None, total=None, mass=None):
    """What each particle senses: an (N, 66) array.

    x: (N, 2) positions, s: (N, 16) states. Particles have mass 1.
    eps is the neighbour radius; eps0 is the radius the network was trained at, used to
    rescale the gradients so a model trained at one radius still reads sensibly at another.
    block: process this many rows of the N x N interaction matrix at a time, to save memory.
    total: the particle count the density gradient is divided by (default N). Set it to the original
    count after cutting particles away, so the survivors sense the same thing they did before.
    mass: (N,) per-particle mass, default 1. A mass of 0 makes a particle invisible to the others, which
    is how the experiments delete particles without changing array sizes.
    """
    n = x.shape[0]
    x = lax.stop_gradient(x)               # the network does not backpropagate through positions
    bs = n if block is None else block
    assert n % bs == 0, "block must divide the particle count"
    xb = x.reshape(n // bs, bs, 2)

    def offsets(xi):                        # xi: (bs, 2) -> r (bs, N, 2), d2 (bs, N)
        r = x[None, :, :] - xi[:, None, :]
        return r, jnp.sum(r * r, axis=-1)

    # density at every particle (the mass of everyone within eps, weighted by closeness)
    m = jnp.ones(n) if mass is None else mass
    rho = lax.map(lambda xi: jnp.sum(_poly6(offsets(xi)[1], eps) * m[None, :], axis=-1), xb).reshape(n)
    vol = m / jnp.maximum(rho, 1e-12)       # the "volume" each particle stands for (0 for a deleted one)
    vs = vol[:, None] * s
    sb = s.reshape(n // bs, bs, -1)

    def rows(args):
        xi, si = args
        r, d2 = offsets(xi)
        w = _poly6(d2, eps)                 # (bs, N)
        g = _spiky_grad(r, d2, eps)         # (bs, N, 2)
        smooth = w @ vs                                                       # (bs, C)
        # gradient of the state: sum_j v_j (s_j - s_i) * grad W_ij
        grad0 = jnp.einsum("bnd,nc->bcd", g, vs) - si[:, :, None] * jnp.einsum("bnd,n->bd", g, vol)[:, None, :]
        # first-order correction: undo the bias that uneven neighbours put into grad0
        moment = jnp.einsum("n,bna,bnd->bad", vol, r, g)                      # (bs, 2, 2)
        inv, det = _inv2x2(moment)
        inv = jnp.where((det >= DET_MIN)[:, None, None], inv, jnp.eye(2))
        grad1 = jnp.einsum("bcd,bde->bce", grad0, inv)
        # straight-through: use the corrected gradient forward, the plain one for backprop
        grad = grad0 + lax.stop_gradient(grad1 - grad0)
        drho = jnp.sum(g * m[None, :, None], axis=1)                          # (bs, 2)
        return smooth, grad, drho

    smooth, grad, drho = lax.map(rows, (xb, sb))
    smooth = smooth.reshape(n, -1)
    grad = grad.reshape(n, STATE_DIMS, SPATIAL_DIMS)
    drho = drho.reshape(n, SPATIAL_DIMS)

    scale = eps / eps0
    grad = _lognorm(grad * scale).reshape(n, -1)
    drho = _lognorm(drho * scale ** (1 + SPATIAL_DIMS) / (n if total is None else total))   # independent of particle count
    return jnp.concatenate([s, smooth, grad, drho], axis=-1)


# =============================================================================
# 3. One update step
# =============================================================================
def step(params, x, s, update, eps, eps0, block=None, total=None, mass=None):
    """Advance every particle one step.

    update: (N, 1) array of 0/1, which particles actually update this step.
    Returns new positions, new states, and the mean move length (used as a training penalty).
    """
    z = perceive(x, s, eps, eps0, block, total, mass)
    y = jax.nn.relu(z @ params["w1"] + params["b1"]) @ params["w2"]
    dx, ds = y[:, :SPATIAL_DIMS], y[:, SPATIAL_DIMS:]
    length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
    dx = ALPHA * dx * eps / (1.0 + length)          # moves are at most ALPHA * EPS
    move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
    return x + update * dx, s + update * ds, move


def colour(s):
    """The colour a particle shows: its last three state channels, shifted to centre on grey."""
    return s[..., -3:] + 0.5


# =============================================================================
# 4. Seeds, damage and rollouts
# =============================================================================
def seed(key, n, radius=0.2):
    """A uniform disc of particles with all-zero state: the 'egg' the shape grows from."""
    kr, kt = jax.random.split(key)
    r = jnp.sqrt(jax.random.uniform(kr, (n,))) * radius
    t = jax.random.uniform(kt, (n,)) * 2.0 * math.pi
    return jnp.stack([r * jnp.cos(t), r * jnp.sin(t)], axis=-1), jnp.zeros((n, STATE_DIMS))


def damage(x, s, centre, radius):
    """Wipe the state of every particle within `radius` of `centre` (positions are untouched)."""
    hit = jnp.sum((x - centre) ** 2, axis=-1) < radius * radius
    return jnp.where(hit[:, None], 0.0, s)


def rollout(params, key, x, s, steps, eps, eps0, block=None):
    """Run `steps` update steps on one sample. Jit-friendly: `steps` is a static int."""
    def body(carry, k):
        x, s = carry
        update = jax.random.bernoulli(k, UPDATE_PROB, (x.shape[0], 1)).astype(x.dtype)
        x, s, _ = step(params, x, s, update, eps, eps0, block)
        return (x, s), None

    (x, s), _ = lax.scan(body, (x, s), jax.random.split(key, steps))
    return x, s
