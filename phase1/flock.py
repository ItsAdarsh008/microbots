"""Phase 1: a flocking swarm that uses only local rules, written in JAX.

Every bot follows the same three classic "boids" rules, using only the
neighbours it can see within VIEW_RADIUS:
  * separation: don't crowd the bots right next to you
  * alignment:  steer toward the average heading of your neighbours
  * cohesion:   steer toward the average position of your neighbours
Two more forces come from the environment, not from other bots:
  * walls:  bots sense nearby walls and turn away from them
  * beacon: once switched on, every bot can sense the direction of a target
            zone (like a light), and is pulled into it

Finding neighbours is the expensive part. Two methods are included:
  * brute force: every bot checks every other bot (N x N distances)
  * cell list:   space is cut into a grid of cells, and each bot only checks
                 the 9 cells around its own
They give the same result; the cell list is much faster for big swarms.

Run it on its own to print a quick summary:
    python phase1/flock.py
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

# --- The world ---------------------------------------------------------------
BOX = 100.0                        # square world from 0 to BOX in x and y
TARGET = jnp.array([60.0, 50.0])   # centre of the beacon's target zone
TARGET_RADIUS = 30.0               # the zone the beacon gathers bots into

# --- What a bot can sense ----------------------------------------------------
VIEW_RADIUS = 5.0     # neighbours closer than this are "seen"
SEP_RADIUS = 1.5      # seen neighbours closer than this are "too close"
WALL_MARGIN = 8.0     # walls are felt within this distance

# --- How strongly each rule steers (per step) --------------------------------
W_SEPARATION = 0.30
W_ALIGNMENT = 0.08
W_COHESION = 0.01
W_WALL = 0.10
W_BEACON = 0.03

# --- Speed limits ------------------------------------------------------------
MIN_SPEED = 0.3
MAX_SPEED = 1.0

# --- Cell list settings ------------------------------------------------------
GRID = math.ceil(BOX / VIEW_RADIUS)   # 20 x 20 cells, each VIEW_RADIUS wide
CELL_SIZE = BOX / GRID
MAX_PER_CELL = 32                     # how many bots we look at per cell
# The 9 cells to check: your own cell plus the 8 around it.
OFFSETS = jnp.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])


# =============================================================================
# 1. Finding neighbours
# =============================================================================
def brute_force_neighbours(pos):
    """Every bot considers every other bot as a possible neighbour.

    Returns (candidates, valid), both shape (N, N): row i lists the indices bot i
    should look at, and which of those entries are real.
    """
    n = pos.shape[0]
    candidates = jnp.broadcast_to(jnp.arange(n), (n, n))
    valid = jnp.ones((n, n), dtype=bool)
    return candidates, valid


def cell_counts(pos):
    """How many bots are in each grid cell (used by the checks)."""
    cells = jnp.clip(jnp.floor(pos / CELL_SIZE).astype(jnp.int32), 0, GRID - 1)
    cell_id = cells[:, 0] * GRID + cells[:, 1]
    return jnp.bincount(cell_id, length=GRID * GRID)


def cell_list_neighbours(pos):
    """Each bot only considers bots in its own cell and the 8 cells around it.

    Returns (candidates, valid), both shape (N, 9 * MAX_PER_CELL).
    """
    n = pos.shape[0]

    # Which cell is each bot in? Cells are numbered 0 .. GRID*GRID - 1.
    cells = jnp.clip(jnp.floor(pos / CELL_SIZE).astype(jnp.int32), 0, GRID - 1)
    cell_id = cells[:, 0] * GRID + cells[:, 1]

    # Sort the bots by cell, so each cell's bots sit next to each other.
    order = jnp.argsort(cell_id)
    sorted_id = cell_id[order]

    # For every cell, where its bots start and end in the sorted list.
    every_cell = jnp.arange(GRID * GRID)
    start = jnp.searchsorted(sorted_id, every_cell, side="left")
    end = jnp.searchsorted(sorted_id, every_cell, side="right")

    def candidates_for(cell):
        """The bots in the 3 x 3 block of cells around one cell."""
        nearby = cell + OFFSETS                               # (9, 2)
        on_grid = ((nearby >= 0) & (nearby < GRID)).all(axis=1)
        nearby = jnp.clip(nearby, 0, GRID - 1)
        nearby_id = nearby[:, 0] * GRID + nearby[:, 1]        # (9,)

        # Up to MAX_PER_CELL slots per cell: start, start+1, ...
        slot = start[nearby_id][:, None] + jnp.arange(MAX_PER_CELL)[None, :]
        valid = on_grid[:, None] & (slot < end[nearby_id][:, None])
        idx = order[jnp.clip(slot, 0, n - 1)]
        return idx.reshape(-1), valid.reshape(-1)

    return jax.vmap(candidates_for)(cells)


# =============================================================================
# 2. The rules, for ONE bot
# =============================================================================
def wall_push(p):
    """A push away from any wall closer than WALL_MARGIN, stronger when closer."""
    near_low = jnp.maximum(0.0, WALL_MARGIN - p)          # near left / bottom
    near_high = jnp.maximum(0.0, WALL_MARGIN - (BOX - p))  # near right / top
    return (near_low - near_high) / WALL_MARGIN


def steer_one(i, candidates, valid, pos, vel, beacon_on):
    """Work out bot i's steering (a change in velocity) from what it can sense.

    i:          this bot's index
    candidates: indices of bots it might see (from a neighbour finder)
    valid:      which candidate entries are real
    pos, vel:   the whole swarm, shape (N, 2); bot i only reads its candidates
    beacon_on:  whether the target zone's beacon is switched on
    """
    p, v = pos[i], vel[i]
    other_p, other_v = pos[candidates], vel[candidates]

    offset = other_p - p                                   # vectors to each candidate
    dist = jnp.linalg.norm(offset, axis=1)
    seen = valid & (candidates != i) & (dist < VIEW_RADIUS)
    too_close = seen & (dist < SEP_RADIUS)
    count = jnp.maximum(seen.sum(), 1)                     # avoid dividing by 0
    any_seen = seen.any()

    # Separation: push away from bots that are too close, harder when closer.
    push = -offset / jnp.maximum(dist, 1e-3)[:, None] ** 2
    separation = (push * too_close[:, None]).sum(axis=0)

    # Alignment: match the average velocity of the bots you can see.
    avg_vel = (other_v * seen[:, None]).sum(axis=0) / count
    alignment = jnp.where(any_seen, avg_vel - v, 0.0)

    # Cohesion: move toward the average position of the bots you can see.
    cohesion = (offset * seen[:, None]).sum(axis=0) / count

    # Beacon: once switched on, head for the target zone.
    to_target = TARGET - p
    target_dist = jnp.linalg.norm(to_target)
    # Full strength outside the zone, fading toward the centre inside it,
    # so bots settle into the zone instead of flying straight through.
    strength = jnp.minimum(target_dist / TARGET_RADIUS, 1.0)
    direction = to_target / jnp.maximum(target_dist, 1e-6)
    beacon = jnp.where(beacon_on, strength * direction, 0.0)

    return (W_SEPARATION * separation
            + W_ALIGNMENT * alignment
            + W_COHESION * cohesion
            + W_WALL * wall_push(p)
            + W_BEACON * beacon)


# Run steer_one for every bot at once. in_axes says which inputs are per bot
# (0 = split along the first axis) and which are shared by all bots (None).
steer_all = jax.vmap(steer_one, in_axes=(0, 0, 0, None, None, None))


# =============================================================================
# 3. Moving the whole swarm one step
# =============================================================================
def move(pos, vel, steering):
    """Apply steering, keep speed within limits, move, and bounce off walls."""
    vel = vel + steering
    speed = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel / jnp.maximum(speed, 1e-6) * jnp.clip(speed, MIN_SPEED, MAX_SPEED)
    pos = pos + vel

    # Hard walls as a safety net (the wall push usually turns bots first).
    low, high = pos < 0, pos > BOX
    pos = jnp.where(low, -pos, jnp.where(high, 2 * BOX - pos, pos))
    vel = jnp.where(low | high, -vel, vel)
    return pos, vel


def step(pos, vel, beacon_on, use_cells=True):
    """One time step for the whole swarm."""
    finder = cell_list_neighbours if use_cells else brute_force_neighbours
    candidates, valid = finder(pos)
    everyone = jnp.arange(pos.shape[0])
    steering = steer_all(everyone, candidates, valid, pos, vel, beacon_on)
    return move(pos, vel, steering)


def initial_state(key, n_bots):
    """Bots at random spots, each moving in a random direction."""
    k_pos, k_angle = jax.random.split(key)
    pos = jax.random.uniform(k_pos, (n_bots, 2), minval=WALL_MARGIN,
                             maxval=BOX - WALL_MARGIN)
    angle = jax.random.uniform(k_angle, (n_bots,), maxval=2 * jnp.pi)
    vel = 0.6 * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
    return pos, vel


@partial(jax.jit, static_argnames=("n_bots", "n_steps", "beacon_from", "use_cells"))
def simulate(key, n_bots=1000, n_steps=600, beacon_from=300, use_cells=True):
    """Run the swarm and return every frame's positions and velocities.

    The beacon switches on at step `beacon_from`. Returns two arrays of shape
    (n_steps, n_bots, 2).
    """
    pos, vel = initial_state(key, n_bots)

    def tick(carry, t):
        pos, vel = carry
        pos, vel = step(pos, vel, t >= beacon_from, use_cells)
        return (pos, vel), (pos, vel)

    _, (traj_pos, traj_vel) = jax.lax.scan(tick, (pos, vel), jnp.arange(n_steps))
    return traj_pos, traj_vel


# =============================================================================
# 4. Measuring the swarm
# =============================================================================
def local_alignment(pos, vel):
    """How aligned each bot is with the bots it can see, averaged over the swarm.

    0 means headings are random, 1 means every bot moves exactly like its
    neighbours. Uses brute force, so it is exact.
    """
    heading = vel / jnp.maximum(jnp.linalg.norm(vel, axis=1, keepdims=True), 1e-6)
    dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    seen = dist < VIEW_RADIUS                       # includes the bot itself
    mean_heading = (seen[:, :, None] * heading[None, :, :]).sum(1) / seen.sum(1)[:, None]
    return jnp.linalg.norm(mean_heading, axis=1).mean()


def fraction_in_target(pos):
    """Share of bots inside the target zone."""
    return (jnp.linalg.norm(pos - TARGET, axis=1) <= TARGET_RADIUS).mean()


if __name__ == "__main__":
    traj_pos, traj_vel = simulate(jax.random.PRNGKey(0))
    print("trajectory shape:", traj_pos.shape)                      # (600, 1000, 2)
    print("alignment at start:", float(local_alignment(traj_pos[0], traj_vel[0])))
    print("alignment before beacon:", float(local_alignment(traj_pos[299], traj_vel[299])))
    print("share in target at end:", float(fraction_in_target(traj_pos[-1])))
