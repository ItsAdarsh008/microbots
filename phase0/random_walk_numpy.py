"""Phase 0, version 1: a 2D random walk of N bots in plain NumPy.

Every bot wanders inside a square box. Each step it keeps most of its
previous velocity, gets a small random nudge, moves, and bounces off any
wall it crosses.

Run it on its own to print a quick summary:
    python phase0/random_walk_numpy.py
"""

import numpy as np

BOX = 10.0        # the box runs from 0 to BOX in both x and y
MOMENTUM = 0.9    # how much of last step's velocity each bot keeps (0 to 1)
NOISE = 0.05      # how big the random nudge is each step


def reflect(pos, vel, box=BOX):
    """Bounce any bot that crossed a wall back inside, and flip its velocity.

    pos, vel: arrays of shape (N, 2). Works on every bot and both axes at once.
    """
    # Left and bottom walls (position below 0): mirror back across 0.
    hit_low = pos < 0
    pos = np.where(hit_low, -pos, pos)
    vel = np.where(hit_low, -vel, vel)

    # Right and top walls (position above box): mirror back across box.
    hit_high = pos > box
    pos = np.where(hit_high, 2 * box - pos, pos)
    vel = np.where(hit_high, -vel, vel)
    return pos, vel


def step(pos, vel, rng):
    """Advance every bot by one time step. Returns the new pos and vel."""
    nudge = rng.normal(0.0, NOISE, size=pos.shape)  # one random nudge per bot per axis
    vel = MOMENTUM * vel + nudge
    pos = pos + vel
    return reflect(pos, vel)


def simulate(n_bots=100, n_steps=300, seed=0):
    """Run the walk and return every frame as one array of shape (n_steps, n_bots, 2)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, BOX, size=(n_bots, 2))  # random starting spots
    vel = np.zeros((n_bots, 2))                  # everyone starts still

    frames = []
    for _ in range(n_steps):        # this loop is over TIME, never over bots
        pos, vel = step(pos, vel, rng)
        frames.append(pos.copy())
    return np.stack(frames)


if __name__ == "__main__":
    traj = simulate()
    print("trajectory shape:", traj.shape)                      # (300, 100, 2)
    print("all inside box:", bool(((traj >= 0) & (traj <= BOX)).all()))
    print("bot 3 at step 50:", traj[50, 3])
