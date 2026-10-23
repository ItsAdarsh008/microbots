"""Run the Phase 2 checks and print PASS or FAIL for each. Takes a few minutes.

    python phase2/checks.py
"""

from collections import deque

import jax.numpy as jnp
import numpy as np

import assemble as a

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N_MOVERS, STEPS, EVERY = 150, 7000, 20

# 1. The gradient equals the hop count from the seeds (checked by breadth-first search).
pos, est, state, grad, localized, start_time, is_seed = (jnp.asarray(x) for x in a.initial_state(N_MOVERS))
ids = jnp.arange(pos.shape[0])
for t in range(60):                       # before START_DELAY, so nobody moves yet
    grad, _ = a.decide_all(ids, t, pos, state, grad, is_seed)
p = np.asarray(pos)
touching = np.linalg.norm(p[:, None] - p[None], axis=2) < a.G_DIST
hops = np.full(len(p), a.MAX_GRADIENT)
hops[:4] = a.SEED_GRADIENT
queue = deque(range(4))
while queue:                              # breadth-first search out from the seeds
    i = queue.popleft()
    for j in np.where(touching[i])[0]:
        if hops[j] > hops[i] + 1:
            hops[j] = hops[i] + 1
            queue.append(j)
check("gradient equals hop count from the seeds", np.array_equal(np.asarray(grad), hops),
      f"gradients range 0 to {int(np.asarray(grad).max())}")

# 2-7. Assemble the wall and inspect the result.
mask, traj_pos, traj_state, final, seeds = a.run("wall", N_MOVERS, STEPS, EVERY)
pos, est, state, grad, localized, start_time = final
joined = (state == a.JOINED) & ~seeds

check("every mover joins the shape", joined.sum() == N_MOVERS, f"{joined.sum()} of {N_MOVERS}")
cov = a.coverage(mask, pos, state)
check("the shape is at least 85% covered", cov >= 0.85, f"{cov:.0%}")

d = np.linalg.norm(pos[:, None] - pos[None], axis=2) + np.eye(len(pos)) * 99
check("no two robots overlap", d.min() >= 0.85, f"closest pair {d.min():.2f} robot widths apart")

err = np.linalg.norm(est - pos, axis=1)[joined]
check("joined robots know where they are", err.max() < 0.3,
      f"position error mean {err.mean():.3f}, max {err.max():.3f} robot widths")

inside = np.array([a.inside_shape_np(q, mask) for q in pos[joined]])
check("joined robots are inside the shape", inside.mean() >= 0.95, f"{inside.mean():.0%} inside")

moved_after_joining = 0
for f in range(1, traj_pos.shape[0]):
    was_joined = traj_state[f - 1] == a.JOINED
    moved_after_joining += int((np.linalg.norm(traj_pos[f] - traj_pos[f - 1], axis=1)[was_joined] > 1e-6).sum())
check("joined robots never move again", moved_after_joining == 0, f"{moved_after_joining} moves")

busiest = int(((traj_state == a.MOVE_OUT) | (traj_state == a.MOVE_IN)).sum(1).max())
check("movers never exceed MAX_MOVERS", busiest <= a.MAX_MOVERS,
      f"at most {busiest} moving at once, limit {a.MAX_MOVERS}")

# 8. Locality: a moving robot's decision ignores robots outside R_COMM.
mid = traj_pos.shape[0] // 3
m_pos = jnp.asarray(traj_pos[mid])
m_state = jnp.asarray(traj_state[mid])
mover = int(np.where((traj_state[mid] == a.MOVE_OUT) | (traj_state[mid] == a.MOVE_IN))[0][0])
loc_now = (m_state == a.JOINED)                    # joined robots and seeds are localized
est_now = jnp.where(loc_now[:, None], m_pos, 0.0)  # and know their true positions
args = (jnp.asarray(1), m_state, jnp.asarray(np.full(len(pos), 5)), loc_now,
        jnp.zeros(len(pos), jnp.int32), jnp.asarray(seeds), jnp.asarray(mask))


def decision(pos_now):
    g, st, grd, loc, stt, sd, msk = args
    return a.move_one(mover, g, pos_now, est_now, st, grd, loc, stt, sd, msk)


dist = np.linalg.norm(traj_pos[mid] - traj_pos[mid][mover], axis=1)
far = int(np.argmax(dist))
base = decision(m_pos)
moved_far = decision(m_pos.at[far].add(1.0))
same = all(bool(jnp.allclose(x, y)) for x, y in zip(base, moved_far))
check("a robot outside R_COMM doesn't affect a mover", same,
      f"moved robot {far}, {dist[far]:.0f} widths away")
near = int(np.argsort(np.where(dist > 0, dist, np.inf))[0])
moved_near = decision(m_pos.at[near].add(0.3))
changed = not all(bool(jnp.allclose(x, y)) for x, y in zip(base, moved_near))
check("a robot within R_COMM does affect a mover", changed,
      f"nudged robot {near}, {dist[near]:.1f} widths away")

# 9. The other shapes assemble too.
for shape in ("star", "k"):
    mask_s, _, _, final_s, seeds_s = a.run(shape, N_MOVERS, STEPS, EVERY)
    pos_s, _, state_s = final_s[0], final_s[1], final_s[2]
    joined_s = int(((state_s == a.JOINED) & ~seeds_s).sum())
    cov_s = a.coverage(mask_s, pos_s, state_s)
    check(f"{shape} assembles", joined_s == N_MOVERS and cov_s >= 0.85,
          f"{joined_s} of {N_MOVERS} joined, {cov_s:.0%} covered")

print(f"\n{sum(results)}/{len(results)} checks passed")
