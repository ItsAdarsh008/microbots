"""Phase 2: Kilobot-style self-assembly into a shape, written in JAX.

A simplified recreation of Rubenstein, Cornejo & Nagpal, "Programmable
self-assembly in a thousand-robot swarm" (Science, 2014).

Every robot runs the same program and only uses what it can sense nearby:
the distance to each robot within R_COMM, plus a few numbers those robots
broadcast (gradient, state, and position estimate if they have one).

Four "seed" robots sit where the shape starts. Everyone else starts packed in
a cluster next to them. Then, over and over:
  1. Gradient:     every robot computes its hop count from the seeds.
  2. Localization: a moving robot works out where it is from distances to
                   robots that already know where they are (trilateration).
  3. Edge-following: robots on the far edge of the cluster set off, one at a
                   time per neighbourhood, and travel clockwise around the
                   outside of the group.
  4. Joining:      once a moving robot is inside the shape, it stops when it
                   is about to leave the shape, or when it reaches a stopped
                   robot with the same or a higher gradient.

Run it on its own to print a quick summary:
    python phase2/assemble.py --shape wall
"""

import argparse
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# --- Robot sensing and motion (distances are in robot diameters) -------------
G_DIST = 1.5        # robots this close count as neighbours for the gradient
R_COMM = 3.0        # how far a robot can talk to and measure other robots
DESIRED = 1.0       # edge-following distance from the nearest stationary robot
TOO_CLOSE = 0.95    # closer than this, robots push apart
YIELD_DIST = 1.8    # a mover waits while an earlier mover is this close
SPEED = 0.12        # distance moved per step
K_RADIAL = 2.0      # how hard a mover corrects its distance to the edge
START_DELAY = 60    # steps to wait so the gradient can settle first
MAX_GRADIENT = 999  # "infinity" for robots with no route to the seeds yet
LOC_ITERS = 8       # trilateration refinement steps per time step

# --- States ------------------------------------------------------------------
WAIT, MOVE_OUT, MOVE_IN, JOINED = 0, 1, 2, 3
STATE_NAMES = ["waiting", "moving (outside shape)", "moving (inside shape)", "joined"]

# --- Seeds: four robots that know their position and start the gradient -------
SEED_POS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]])
SEED_GRADIENT = np.array([0, 1, 1, 1])

# --- The shape is a grid of yes/no cells covering the area right of the seeds --
RES = 0.25                          # each cell is a quarter of a robot wide
ORIGIN = np.array([-0.6, -0.6])     # bottom-left corner of the grid
GRID_SIZE = 200                     # 200 x 200 cells = 50 x 50 robot widths
AREA_PER_ROBOT = 1.2                # shape area to allow per robot (they don't pack perfectly)


# =============================================================================
# 1. Shapes
# =============================================================================
def _cell_centres():
    """x and y of every grid cell's centre, each shape (GRID_SIZE, GRID_SIZE)."""
    c = ORIGIN[:, None] + RES * (np.arange(GRID_SIZE) + 0.5)[None, :]
    return np.meshgrid(c[0], c[1], indexing="ij")


def _wall(x, y, s):
    """A rectangle 2.2 times as wide as it is tall."""
    return (x <= -0.5 + 2.2 * s) & (y <= -0.5 + s)


def _star(x, y, s):
    """A five-pointed star, with one arm pointing back at the seeds."""
    toward_seeds = np.array([-1.0, -1.0]) / np.sqrt(2)
    centre = np.array([0.75, 0.43]) - 0.55 * s * toward_seeds
    angle = np.arctan2(y - centre[1], x - centre[0])
    radius = np.hypot(x - centre[0], y - centre[1])
    first_tip = np.arctan2(toward_seeds[1], toward_seeds[0])
    # Distance from the nearest tip direction, folded into one arm (0..36 degrees).
    off = np.abs((angle - first_tip + np.pi / 5) % (2 * np.pi / 5) - np.pi / 5)
    # Edge of the star: straight line from the tip (radius s) to the inner corner.
    inner = 0.45 * s
    edge = s * inner * np.sin(np.pi / 5) / (
        s * np.sin(off) + inner * np.sin(np.pi / 5 - off) + 1e-9)
    return radius <= edge


def _letter_k(x, y, s):
    """A capital K: a stem plus two thick diagonal arms."""
    stem_w, height = 0.32 * s, s
    left, bottom = -0.5, -0.5
    stem = (x <= left + stem_w) & (y <= bottom + height)

    def bar(x0, y0, x1, y1, thick):
        # Points within thick/2 of the segment from (x0, y0) to (x1, y1).
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        return np.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= thick / 2

    mid_y = bottom + 0.5 * height
    arm_end = left + 0.75 * s
    thick = 0.27 * s
    upper = bar(left + stem_w * 0.6, mid_y, arm_end, bottom + height - thick / 2, thick)
    lower = bar(left + stem_w * 0.6, mid_y, arm_end, bottom + thick / 2, thick)
    return stem | ((upper | lower) & (y <= bottom + height) & (y >= bottom))


SHAPES = {"wall": _wall, "star": _star, "k": _letter_k}


def make_shape(name, n_movers):
    """Build the shape's yes/no grid, sized so n_movers robots can fill it."""
    x, y = _cell_centres()
    quadrant = (x >= -0.6) & (y >= -0.6)   # keep clear of the starting cluster
    target_area = (n_movers + len(SEED_POS)) * AREA_PER_ROBOT

    low, high = 1.0, 60.0                  # binary search for the right scale
    for _ in range(40):
        s = (low + high) / 2
        area = (SHAPES[name](x, y, s) & quadrant).sum() * RES * RES
        low, high = (s, high) if area < target_area else (low, s)
    mask = SHAPES[name](x, y, s) & quadrant
    seeds_in = all(inside_shape_np(p, mask) for p in SEED_POS)
    assert seeds_in, f"shape {name!r} must contain the seeds"
    return mask


def inside_shape_np(p, mask):
    ix, iy = np.floor((np.asarray(p) - ORIGIN) / RES).astype(int)
    return 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE and bool(mask[ix, iy])


def inside_shape(p, mask):
    """Is point p inside the shape? Works inside jit and vmap."""
    idx = jnp.floor((p - ORIGIN) / RES).astype(jnp.int32)
    on_grid = jnp.all((idx >= 0) & (idx < GRID_SIZE))
    idx = jnp.clip(idx, 0, GRID_SIZE - 1)
    return on_grid & mask[idx[0], idx[1]]


# =============================================================================
# 2. The starting layout
# =============================================================================
def initial_state(n_movers):
    """Seeds plus a tightly packed cluster of movers to their left.

    Returns pos, est, state, grad, localized, start_time, is_seed (NumPy arrays).
    """
    cluster = []
    row = 1
    while len(cluster) < n_movers:          # hexagonal rows, top row first
        cols = math.ceil(math.sqrt(n_movers * 1.2))
        for c in range(cols):
            x = -1.0 - c - (0.5 if row % 2 else 0.0)
            cluster.append([x, 0.866 * row])
            if len(cluster) == n_movers:
                break
        row -= 1
    pos = np.concatenate([SEED_POS, np.array(cluster)])
    n = len(pos)
    is_seed = np.arange(n) < len(SEED_POS)

    est = np.where(is_seed[:, None], pos, 0.0)              # only seeds know where they are
    state = np.where(is_seed, JOINED, WAIT)
    grad = np.full(n, MAX_GRADIENT)
    grad[: len(SEED_POS)] = SEED_GRADIENT
    localized = is_seed.copy()
    start_time = np.full(n, 10**9)
    return pos, est, state, grad, localized, start_time, is_seed


# =============================================================================
# 3. Localization: where am I, from distances to robots that know where they are?
# =============================================================================
def residual(p, refs, ref_est, dist):
    """How badly guess p disagrees with the measured distances (0 = perfect)."""
    err = jnp.linalg.norm(p - ref_est, axis=1) - dist
    return (refs * err ** 2).sum() / jnp.maximum(refs.sum(), 1)


def refine(p, refs, ref_est, dist):
    """Nudge guess p so its distance to each reference matches the measurement."""
    for _ in range(LOC_ITERS):
        vec = p - ref_est
        length = jnp.maximum(jnp.linalg.norm(vec, axis=1), 1e-6)
        # Where each reference "says" I should be: along the same line, at the measured distance.
        target = ref_est + vec / length[:, None] * dist[:, None]
        p = p + 0.5 * ((target - p) * refs[:, None]).sum(0) / jnp.maximum(refs.sum(), 1)
    return p


def localize(previous, refs, ref_est, dist):
    """Trilateration from several starting guesses; keep the best fit.

    Starting guesses: the previous estimate, plus points around the references'
    centre (so a robot that has never been localized can still find itself).
    """
    centre = (ref_est * refs[:, None]).sum(0) / jnp.maximum(refs.sum(), 1)
    angles = jnp.arange(8) * (2 * jnp.pi / 8)
    ring = centre + 2.5 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    guesses = jnp.concatenate([previous[None, :], ring])
    refined = jax.vmap(refine, in_axes=(0, None, None, None))(guesses, refs, ref_est, dist)
    errors = jax.vmap(residual, in_axes=(0, None, None, None))(refined, refs, ref_est, dist)
    return refined[jnp.argmin(errors)]


# =============================================================================
# 4. One robot's program for one time step, in two parts
# =============================================================================
# Part A (every robot): update my gradient and decide whether to start moving.
# Part B (moving robots only): localize, move along the edge, decide whether to stop.
# On real robots both parts run on every robot's own chip. In the simulation,
# part B is expensive (trilateration), so it only runs for the robots that are
# actually moving: at most MAX_MOVERS of them per step.
MAX_MOVERS = 24


def decide_one(i, t, pos, state, grad, is_seed):
    """Part A for robot i: returns (my new gradient, whether I start moving now)."""
    n = pos.shape[0]
    ids = jnp.arange(n)
    dist = jnp.linalg.norm(pos - pos[i], axis=1)   # measured distances (like the Kilobots' IR)
    others = ids != i
    in_comm = others & (dist < R_COMM)
    near = others & (dist < G_DIST)
    moving = (state == MOVE_OUT) | (state == MOVE_IN)

    # Gradient: one more than the smallest gradient among touching neighbours.
    smallest = jnp.min(jnp.where(near, grad, MAX_GRADIENT))
    g = jnp.where(is_seed[i], grad[i], jnp.minimum(smallest + 1, MAX_GRADIENT))

    # Start moving? Only if I have the highest gradient among the waiting robots
    # around me (ties: higher id) and nobody nearby is already moving.
    rivals = in_comm & (state == WAIT)
    beats = (g > grad) | ((g == grad) & (i > ids))
    highest = jnp.all(jnp.where(rivals, beats, True))
    quiet = ~jnp.any(in_comm & moving)
    start = (state[i] == WAIT) & ~is_seed[i] & (t >= START_DELAY) & highest & quiet
    return g, start


def move_one(i, g, pos, est, state, grad, localized, start_time, is_seed, mask):
    """Part B for a moving robot i (g is its gradient from part A).

    Returns robot i's new (pos, est, state, localized).
    """
    n = pos.shape[0]
    ids = jnp.arange(n)
    p = pos[i]
    dist = jnp.linalg.norm(pos - p, axis=1)
    others = ids != i
    in_comm = others & (dist < R_COMM)
    near = others & (dist < G_DIST)
    moving = (state == MOVE_OUT) | (state == MOVE_IN)

    # Localization: needs at least 3 stationary, localized robots in range.
    refs = in_comm & ~moving & localized
    can_localize = refs.sum() >= 3
    est_i = jnp.where(can_localize, localize(est[i], refs, est, dist), est[i])
    loc_i = localized[i] | can_localize

    # Edge-following: circle clockwise around the nearest stationary robot,
    # keeping DESIRED distance from it.
    anchor_dist = jnp.where(in_comm & ~moving, dist, jnp.inf)
    j = jnp.argmin(anchor_dist)
    d_j = anchor_dist[j]
    has_anchor = jnp.isfinite(d_j)
    away = (p - pos[j]) / jnp.maximum(d_j, 1e-6)          # unit vector from anchor to me
    clockwise = jnp.array([away[1], -away[0]])            # 90 degrees to the right of "away"
    v = clockwise + K_RADIAL * (DESIRED - jnp.where(has_anchor, d_j, DESIRED)) * away
    crowding = others & (dist < TOO_CLOSE)
    v = v + ((p - pos) / jnp.maximum(dist, 1e-6)[:, None] * crowding[:, None]).sum(0)
    v = SPEED * v / jnp.maximum(jnp.linalg.norm(v), 1e-6)
    v = jnp.where(has_anchor, v, 0.0)

    # Queue politely: wait while a robot that started earlier is close by.
    earlier = moving & ((start_time < start_time[i]) | ((start_time == start_time[i]) & (ids < i)))
    blocked = jnp.any(others & earlier & (dist < YIELD_DIST))

    # Enter the shape once localized and inside it.
    entered = (state[i] == MOVE_OUT) & loc_i & inside_shape(est_i, mask)
    new_state = jnp.where(entered, MOVE_IN, state[i])

    # Inside the shape, stop if the next step would leave it, or if I'm touching
    # a stopped robot whose gradient is the same as mine or higher.
    leaving = ~inside_shape(est_i + v, mask)
    stopped_nearby = near & ((state == JOINED) | is_seed)
    caught_up = jnp.any(stopped_nearby & (grad >= g))
    stop = (new_state == MOVE_IN) & (leaving | caught_up)
    new_state = jnp.where(stop, JOINED, new_state)

    step_now = ~stop & ~blocked
    new_p = jnp.where(step_now, p + v, p)
    new_est = jnp.where(step_now, est_i + v, est_i)   # dead-reckoning guess until next fix
    return new_p, new_est, new_state, loc_i


decide_all = jax.vmap(decide_one, in_axes=(0, None, None, None, None, None))
move_some = jax.vmap(move_one, in_axes=(0, 0, None, None, None, None, None, None, None, None))


def step(t, pos, est, state, grad, localized, start_time, is_seed, mask):
    """One time step for the whole swarm."""
    ids = jnp.arange(pos.shape[0])
    moving = (state == MOVE_OUT) | (state == MOVE_IN)

    # Part A for everyone.
    g, start = decide_all(ids, t, pos, state, grad, is_seed)

    # Part B for up to MAX_MOVERS moving robots. Sorting puts the moving ones first.
    slots = jnp.argsort(~moving)[:MAX_MOVERS]
    real = moving[slots]
    new_pos, new_est, new_state, new_loc = move_some(
        slots, g[slots], pos, est, state, grad, localized, start_time, is_seed, mask)
    # Slots that aren't real movers keep their old values.
    new_pos = jnp.where(real[:, None], new_pos, pos[slots])
    new_est = jnp.where(real[:, None], new_est, est[slots])
    new_state = jnp.where(real, new_state, state[slots])
    new_loc = jnp.where(real, new_loc, localized[slots])
    pos = pos.at[slots].set(new_pos)
    est = est.at[slots].set(new_est)
    state = state.at[slots].set(new_state)
    localized = localized.at[slots].set(new_loc)

    # Robots that decided to start become movers from the next step.
    state = jnp.where(start, MOVE_OUT, state)
    start_time = jnp.where(start, t, start_time)
    return pos, est, state, g, localized, start_time


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def simulate(mask, pos, est, state, grad, localized, start_time, is_seed,
             n_steps=7000, save_every=20):
    """Run the assembly. Returns positions and states every `save_every` steps,
    plus the final full state."""

    def tick(carry, t):
        return step(t, *carry, is_seed, mask), None

    def chunk(carry, c):
        carry, _ = jax.lax.scan(tick, carry, c * save_every + jnp.arange(save_every))
        return carry, (carry[0], carry[2])          # save positions and states

    carry = (pos, est, state, grad, localized, start_time)
    final, (traj_pos, traj_state) = jax.lax.scan(chunk, carry, jnp.arange(n_steps // save_every))
    return traj_pos, traj_state, final


def run(shape="wall", n_movers=150, n_steps=7000, save_every=20):
    """Convenience wrapper: build everything, simulate, return NumPy arrays."""
    mask = make_shape(shape, n_movers)
    init = initial_state(n_movers)
    *state0, is_seed = [jnp.asarray(a) for a in init]
    traj_pos, traj_state, final = simulate(jnp.asarray(mask), *state0, jnp.asarray(is_seed),
                                           n_steps=n_steps, save_every=save_every)
    final = [np.asarray(a) for a in final]
    return mask, np.asarray(traj_pos), np.asarray(traj_state), final, init[-1]


# =============================================================================
# 5. Measuring the result
# =============================================================================
def coverage(mask, pos, state):
    """Share of the shape's area that has a joined robot within 0.75 of it."""
    x, y = _cell_centres()
    cells = np.stack([x[mask], y[mask]], axis=1)
    joined = pos[state == JOINED]
    d = np.linalg.norm(cells[:, None, :] - joined[None, :, :], axis=2).min(axis=1)
    return float((d <= 0.75).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="wall", choices=list(SHAPES))
    ap.add_argument("--movers", type=int, default=150)
    ap.add_argument("--steps", type=int, default=7000)
    a = ap.parse_args()
    mask, traj_pos, traj_state, final, is_seed = run(a.shape, a.movers, a.steps)
    pos, est, state, grad, localized, start_time = final
    print("joined:", int((state == JOINED).sum() - is_seed.sum()), "of", a.movers)
    print("still moving:", int(((state == MOVE_OUT) | (state == MOVE_IN)).sum()))
    print("shape coverage:", round(coverage(mask, pos, state), 3))
    err = np.linalg.norm(est - pos, axis=1)[(state == JOINED) & ~is_seed]
    print("localization error (mean, max):", round(float(err.mean()), 3), round(float(err.max()), 3))
