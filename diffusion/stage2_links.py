"""Stage 2 version 1: can the finished swarm be linked into a rigid structure with the v1 bot's connectors?
Resource model from assets/microbot_v1/DESIGN.md: each bot has 2 arm tips and 4 equatorial ports. A short link (centres within 1.2 bodies,
i.e. touching) uses one connector at each end (a port, or a tip if no port is left). A long link (up to the arm reach, 2.35 bodies) needs an
arm tip at one end (tip to port) or at both ends (tip to tip). Greedy: shortest candidate links first, as long as both ends have a connector.
Rigidity: a 2D pin-jointed frame of N bots is rigid iff the rank of its rigidity matrix is 2N - 3.
    python stage2_links.py runs/<name>"""
import json, sys
import numpy as np, torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("stage2_eval.py").read().split("print(f\"{run}: no retraining")[0])
import os
SHORT, REACH = 1.2, float(os.environ.get('REACH', 2.35))


def link(X, s):
    N = len(X); d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); iu = np.triu_indices(N, 1)
    cand = [(d[i, j], i, j) for i, j in zip(*iu) if d[i, j] <= REACH * s]; cand.sort()
    ports, tips = np.full(N, 4), np.full(N, 2); edges = []
    for L, i, j in cand:
        if L <= SHORT * s:                                 # touching: port to port (a tip may stand in)
            ci = 'p' if ports[i] else ('t' if tips[i] else None); cj = 'p' if ports[j] else ('t' if tips[j] else None)
            if ci and cj:
                (ports if ci == 'p' else tips)[i] -= 1; (ports if cj == 'p' else tips)[j] -= 1; edges.append((i, j))
        else:                                              # long: a tip at one end, a port (or tip) at the other
            for a, b in ((i, j), (j, i)) if tips[i] >= tips[j] else ((j, i), (i, j)):
                if tips[a] and (ports[b] or tips[b]):
                    tips[a] -= 1; (ports if ports[b] else tips)[b] -= 1; edges.append((i, j)); break
    return edges, d


def rigidity(X, edges):
    N = len(X); R = np.zeros((len(edges), 2 * N))
    for k, (i, j) in enumerate(edges):
        v = X[i] - X[j]; R[k, 2 * i:2 * i + 2] = v; R[k, 2 * j:2 * j + 2] = -v
    return np.linalg.matrix_rank(R, tol=1e-6) if len(edges) else 0


for K, vmax in ((32, 0.5),):
    X, path = go(K, vmax, 1.0)
    res = []
    for i in range(24):
        s = sp[i, 0, 0].item(); edges, d = link(X[i], s)
        A = np.zeros((256, 256), bool)
        for a, b in edges: A[a, b] = A[b, a] = True
        k, lab = connected_components(csr_matrix(A)); big = np.bincount(lab).max() / 256
        rank = rigidity(X[i], edges); res.append((big, 2 * len(edges) / 256, rank / (2 * 256 - 3), len(edges)))
    r = np.mean(res, 0)
    print(f"{run}: reach {REACH} bodies; rounds {K}, max step {vmax}, bodies touching; links limited to 4 ports + 2 tips per bot (24 val shapes)\n"
          f"   largest connected piece {r[0]:.2f} | mean links per bot {r[1]:.2f} | rigidity rank / (2N-3) = {r[2]:.2f} (1.00 = fully rigid) | links {r[3]:.0f}")
