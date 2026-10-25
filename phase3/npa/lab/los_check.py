"""How many neighbours can a bot see at 1x and 2x the sensing radius if bodies block the line of sight?

A swarm of hard discs (body radius r) is packed at the density where an average bot has about 7 neighbours within eps
(random sequential addition, a disc in a square patch; only bots away from the edge are counted). A neighbour is visible
if the straight segment between the two centres does not cross another body. This is a geometric stand-in for an optical
or line-of-sight sensor; a radio link relayed through neighbours would not be blocked. Run: python lab/los_check.py
"""
import numpy as np

EPS, R, NB = 0.05, 0.0129, 7.0          # sensing radius (world units), body radius (15 mm / 1160 mm per unit), target neighbours within eps


def swarm(rng, side, rho):
    n_target = int(rho * side * side)
    pts = np.empty((0, 2))
    tries = 0
    while len(pts) < n_target and tries < 400000:
        c = rng.uniform(0, side, (2000, 2)); tries += 2000
        for p in c:
            if len(pts) == 0 or np.min(np.hypot(*(pts - p).T)) >= 2 * R:
                pts = np.vstack([pts, p])
                if len(pts) >= n_target:
                    break
    return pts


def visible(pts, i, j):
    a, b = pts[i], pts[j]
    ab = b - a
    L2 = ab @ ab
    t = np.clip(((pts - a) @ ab) / L2, 0, 1)
    d = np.hypot(*(pts - (a + t[:, None] * ab)).T)
    d[[i, j]] = 9
    return not np.any(d < R)


def run(seed, side=1.0):
    rng = np.random.default_rng(seed)
    rho = NB / (np.pi * EPS ** 2)
    pts = swarm(rng, side, rho)
    D = np.hypot(*(pts[:, None] - pts[None]).T)
    core = np.where((pts > 3 * EPS).all(1) & (pts < side - 3 * EPS).all(1))[0]
    out = {1: [], 2: []}
    tot = {1: [], 2: []}
    for i in core:
        for k in (1, 2):
            nb = np.where((D[i] < k * EPS) & (D[i] > 0))[0]
            tot[k].append(len(nb)); out[k].append(sum(visible(pts, i, j) for j in nb))
    return len(pts), len(core), {k: (np.mean(tot[k]), np.mean(out[k])) for k in (1, 2)}


if __name__ == "__main__":
    for s in range(3):
        n, c, r = run(s)
        print(f"seed {s}: {n} bots, {c} counted | within eps: {r[1][0]:.1f} in range, {r[1][1]:.1f} visible | within 2 eps: {r[2][0]:.1f} in range, {r[2][1]:.1f} visible")
