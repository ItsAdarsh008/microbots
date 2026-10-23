"""Run the Phase 1 checks from the roadmap and print PASS or FAIL for each.

    python phase1/checks.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import flock

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


key = jax.random.PRNGKey(0)
traj_pos, traj_vel = flock.simulate(key, 1000, 600, 300)
traj_pos, traj_vel = np.asarray(traj_pos), np.asarray(traj_vel)

# 1. Shapes and limits.
check("trajectory shape is (600, 1000, 2)", traj_pos.shape == (600, 1000, 2), str(traj_pos.shape))
inside = ((traj_pos >= 0) & (traj_pos <= flock.BOX)).all()
check("every bot stays inside the world", inside)
speed = np.linalg.norm(traj_vel, axis=2)
check("speeds stay within the limits",
      (speed >= flock.MIN_SPEED - 1e-4).all() and (speed <= flock.MAX_SPEED + 1e-4).all(),
      f"{speed.min():.2f} to {speed.max():.2f}")

# 2. The cell list never has more bots in a cell than it can look at.
busiest = max(int(flock.cell_counts(jnp.asarray(traj_pos[t])).max()) for t in range(600))
check("no cell ever overflows MAX_PER_CELL", busiest <= flock.MAX_PER_CELL,
      f"busiest cell held {busiest} of {flock.MAX_PER_CELL}")

# 3. Cell list and brute force give the same steering.
pos, vel = jnp.asarray(traj_pos[200]), jnp.asarray(traj_vel[200])
everyone = jnp.arange(pos.shape[0])
steer_cells = flock.steer_all(everyone, *flock.cell_list_neighbours(pos), pos, vel, True)
steer_brute = flock.steer_all(everyone, *flock.brute_force_neighbours(pos), pos, vel, True)
gap = float(jnp.abs(steer_cells - steer_brute).max())
check("cell list matches brute force", gap < 1e-4, f"largest difference {gap:.1e}")

# 4. Locality: a bot's steering only depends on bots within VIEW_RADIUS.
#    Move a far-away bot and bot 0's steering must not change at all.
p0 = pos[0]
dist = jnp.linalg.norm(pos - p0, axis=1)
far = int(jnp.argmax(dist))
moved = pos.at[far].set(jnp.clip(pos[far] + 3.0, 0, flock.BOX))
before = flock.steer_one(0, everyone, jnp.ones(len(pos), bool), pos, vel, False)
after = flock.steer_one(0, everyone, jnp.ones(len(pos), bool), moved, vel, False)
check("a far-away bot doesn't affect bot 0", bool(jnp.allclose(before, after)),
      f"moved bot {far}, {float(dist[far]):.0f} units away")
#    ...but a bot inside VIEW_RADIUS does.
near = int(jnp.argsort(jnp.where(dist > 0, dist, jnp.inf))[0])
nudged = pos.at[near].set(pos[near] + 0.5)
after_near = flock.steer_one(0, everyone, jnp.ones(len(pos), bool), nudged, vel, False)
check("a nearby bot does affect bot 0", not bool(jnp.allclose(before, after_near)),
      f"nudged bot {near}, {float(dist[near]):.1f} units away")

# 5. Flocking emerges: headings line up with neighbours, without a leader.
start = float(flock.local_alignment(jnp.asarray(traj_pos[0]), jnp.asarray(traj_vel[0])))
flocked = float(flock.local_alignment(jnp.asarray(traj_pos[299]), jnp.asarray(traj_vel[299])))
check("bots line up with their neighbours", flocked > 0.7 and flocked > start + 0.3,
      f"alignment {start:.2f} at start, {flocked:.2f} after 300 steps")

# 6. The beacon gathers the swarm.
share = float(flock.fraction_in_target(jnp.asarray(traj_pos[-1])))
check("most bots end up in the target zone", share > 0.6, f"{share:.0%} inside")

# 7. Speed: steps per second for 1,000 bots, and brute force vs cell list.
def steps_per_second(n, use_cells, repeats=30):
    pos, vel = flock.initial_state(jax.random.PRNGKey(1), n)
    fast = jax.jit(flock.step, static_argnames="use_cells")
    fast(pos, vel, False, use_cells=use_cells)[0].block_until_ready()   # compile
    start = time.perf_counter()
    for _ in range(repeats):
        pos, vel = fast(pos, vel, False, use_cells=use_cells)
    pos.block_until_ready()
    return repeats / (time.perf_counter() - start)


cells_1k = steps_per_second(1000, True)
check("1,000 bots run at 30+ steps per second", cells_1k > 30, f"{cells_1k:.0f} steps/s")
for n in (1000, 5000):
    brute, cells = steps_per_second(n, False), steps_per_second(n, True)
    print(f"       {n:>5} bots: brute force {brute:6.0f} steps/s, cell list {cells:6.0f} steps/s")

print(f"\n{sum(results)}/{len(results)} checks passed")
