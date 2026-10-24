"""Ways to hurt a swarm, shared by the evaluator and by training.

Every function works on one swarm (x: (N, 2), s: (N, 16), mem: (N, M), alive: (N,) of 0/1) and
returns the damaged (x, s, mem, alive). `level` is a number whose meaning depends on the kind:

  wipe     zero the state and memory within `level` of a random live particle (what the paper trains on)
  cut      delete the particles within `level` of that particle, for good
  drag     shift the particles within 0.2 of it by `level` in a random direction, tearing them off
  clump    pull the particles within 0.4 of it toward it, by the fraction `level`
  shuffle  scramble the states of the particles within `level` of it (wrong information, not none)
  noise    jostle every live particle: positions by level * eps / 2, states by level
"""

import jax
import jax.numpy as jnp

KINDS = ("wipe", "cut", "drag", "clump", "shuffle", "noise")
DRAG_PATCH = 0.2
CLUMP_PATCH = 0.4


def apply(kind, level, key, x, s, mem, alive, eps):
    n = x.shape[0]
    k_centre, k_dir, k_perm, k_noise = jax.random.split(key, 4)
    centre = x[jax.random.categorical(k_centre, jnp.log(alive + 1e-9))]
    d2 = jnp.sum((x - centre) ** 2, axis=-1)
    live = alive > 0

    if kind == "wipe":
        hit = (d2 < level**2) & live
        return x, jnp.where(hit[:, None], 0.0, s), jnp.where(hit[:, None], 0.0, mem), alive
    if kind == "cut":
        return x, s, mem, jnp.where(d2 < level**2, 0.0, alive)
    if kind == "drag":
        hit = (d2 < DRAG_PATCH**2) & live
        angle = jax.random.uniform(k_dir) * 2.0 * jnp.pi
        shift = level * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        return jnp.where(hit[:, None], x + shift, x), s, mem, alive
    if kind == "clump":
        hit = (d2 < CLUMP_PATCH**2) & live
        return jnp.where(hit[:, None], centre + (1.0 - level) * (x - centre), x), s, mem, alive
    if kind == "shuffle":
        hit = (d2 < level**2) & live
        order = jnp.arange(n)
        hit_first = jnp.argsort(jnp.where(hit, order, n + order))
        scrambled = jnp.argsort(jnp.where(hit, jax.random.uniform(k_perm, (n,)), 2.0 + order))
        src = jnp.where(hit, jnp.zeros(n, jnp.int32).at[hit_first].set(scrambled.astype(jnp.int32)), order)
        return x, s[src], mem[src], alive
    if kind == "noise":
        kx, ks = jax.random.split(k_noise)
        return (x + live[:, None] * jax.random.normal(kx, x.shape) * level * eps * 0.5,
                s + live[:, None] * jax.random.normal(ks, s.shape) * level, mem, alive)
    raise ValueError(f"unknown damage kind {kind}")
