"""Phase 0, version 2: the same random walk, rewritten in JAX.

Same behaviour as random_walk_numpy.py, but written the way the rest of the
project will be:
  * step_one() describes what ONE bot does in one step,
  * jax.vmap turns it into step_all(), which runs it on every bot at once,
  * jax.lax.scan repeats the step n_steps times inside compiled code,
  * jax.jit compiles the whole simulation for speed.

Run it on its own to print a quick summary:
    python phase0/random_walk_jax.py
"""

from functools import partial

import jax
import jax.numpy as jnp

BOX = 10.0        # the box runs from 0 to BOX in both x and y
MOMENTUM = 0.9    # how much of last step's velocity each bot keeps (0 to 1)
NOISE = 0.05      # how big the random nudge is each step


def reflect(pos, vel, box=BOX):
    """Bounce a bot that crossed a wall back inside, and flip its velocity.

    Written with jnp.where instead of `if`, so it works inside jit and vmap.
    """
    hit_low = pos < 0
    pos = jnp.where(hit_low, -pos, pos)
    vel = jnp.where(hit_low, -vel, vel)

    hit_high = pos > box
    pos = jnp.where(hit_high, 2 * box - pos, pos)
    vel = jnp.where(hit_high, -vel, vel)
    return pos, vel


def step_one(pos, vel, key):
    """Move ONE bot by one step.

    pos, vel: shape (2,)  -- this bot's x, y position and velocity
    key:      this bot's own random key for this step
    """
    nudge = NOISE * jax.random.normal(key, shape=(2,))
    vel = MOMENTUM * vel + nudge
    pos = pos + vel
    return reflect(pos, vel)


# vmap: "run step_one on every row". Inputs go from (2,) to (N, 2) and keys to (N,).
step_all = jax.vmap(step_one)


@partial(jax.jit, static_argnames=("n_bots", "n_steps"))
def simulate(key, n_bots=100, n_steps=300):
    """Run the walk and return every frame as one array of shape (n_steps, n_bots, 2).

    n_bots and n_steps are "static": they decide array shapes, so JAX must know
    them while compiling. Changing them triggers a fresh compile.
    """
    key, start_key = jax.random.split(key)
    pos = jax.random.uniform(start_key, (n_bots, 2), minval=0.0, maxval=BOX)
    vel = jnp.zeros((n_bots, 2))

    def tick(carry, _):
        """One time step for the whole swarm. carry = everything that changes."""
        pos, vel, key = carry
        key, step_key = jax.random.split(key)          # fresh key every step
        bot_keys = jax.random.split(step_key, n_bots)  # one key per bot
        pos, vel = step_all(pos, vel, bot_keys)
        return (pos, vel, key), pos                    # (new carry, frame to save)

    _, traj = jax.lax.scan(tick, (pos, vel, key), None, length=n_steps)
    return traj


if __name__ == "__main__":
    traj = simulate(jax.random.PRNGKey(0))
    print("devices:", jax.devices())
    print("trajectory shape:", traj.shape)                      # (300, 100, 2)
    print("all inside box:", bool(((traj >= 0) & (traj <= BOX)).all()))
    print("bot 3 at step 50:", traj[50, 3])
