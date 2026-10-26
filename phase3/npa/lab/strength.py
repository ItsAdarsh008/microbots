"""Geometric strength of a docked swarm, from the contact graph (ROBOT_RESEARCH.md section 4).

    bonds = contact_bonds(pos, radius)               # pairs of bots that count as docked (assumption: you choose the radius)
    Q = q_rel(pos, bonds, ref_pos, ref_bonds)         # relative worst-mode stiffness, 1 = as stiff as the reference, 0 = floppy or split
    rigid_fraction(pos, bonds), critical_fraction(bonds, n), mean_degree(bonds, n)

Each bot is a rigid body with 3 freedoms (u, v, phi). A bond is a spring set (normal kn, tangential kt, rotational kr) at the bond
midpoint; the stiffness matrix K sums them. Its 4th-smallest eigenvalue (after the 3 rigid-body motions) is the softest way the
structure can deform; a disconnected or floppy structure has more zero eigenvalues, so that value is 0.
Units: millimetres and newtons. Defaults follow the research notes (EP-Face shear about 0.4 of normal; rotational stiffness is a rough guess).
"""
import numpy as np

BODY_D_MM = 30.0


def contact_bonds(pos, radius):
    """Pairs (i<j) with centre distance below `radius` (same units as pos)."""
    pos = np.asarray(pos, float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    i, j = np.where(np.triu(d < radius, 1))
    return np.stack([i, j], axis=1)


def stiffness_matrix(pos, bonds, kn=1.0, kt=0.4, kr=None):
    pos = np.asarray(pos, float)
    n = len(pos)
    kr = kn * (BODY_D_MM / 2) ** 2 / 4 if kr is None else kr
    K = np.zeros((3 * n, 3 * n))
    for i, j in bonds:
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        e = d / L
        t = np.array([-e[1], e[0]])
        B = np.zeros((3, 6))
        B[0, [0, 1]] = -e; B[0, [3, 4]] = e                      # normal stretch
        B[1, [0, 1]] = -t; B[1, [3, 4]] = t                      # tangential slip
        B[1, 2] = -L / 2; B[1, 5] = -L / 2                       # including body rotations
        B[2, 2] = -1; B[2, 5] = 1                                # twist
        Ke = B.T @ np.diag([kn, kt, kr]) @ B
        idx = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(idx, idx)] += Ke
    return K


def worst_mode(pos, bonds, **kw):
    """Smallest eigenvalue of K above the 3 rigid-body motions (index 3 of the sorted spectrum)."""
    K = stiffness_matrix(pos, bonds, **kw)
    return float(np.linalg.eigvalsh(K)[3])


def q_rel(pos, bonds, ref_pos, ref_bonds, **kw):
    ref = worst_mode(ref_pos, ref_bonds, **kw)
    return worst_mode(pos, bonds, **kw) / ref if ref > 1e-12 else float("nan")


def components(n, bonds):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in bonds:
        parent[find(i)] = find(j)
    roots = np.array([find(a) for a in range(n)])
    return roots


def largest_component_fraction(n, bonds):
    roots = components(n, bonds)
    return float(np.bincount(roots).max() / n)


def floppy_modes(pos, bonds, tol=1e-7, **kw):
    """Number of near-zero eigenvalues of K beyond the 3 rigid-body motions of one connected piece (rigid-bond model)."""
    K = stiffness_matrix(pos, bonds, **kw)
    w = np.linalg.eigvalsh(K)
    scale = w.max()
    zeros = int((w < tol * scale).sum())
    roots = components(len(pos), bonds)
    pieces = len(np.unique(roots))
    return zeros - 3 * pieces, pieces


def rigid_fraction(pos, bonds, **kw):
    """Fraction of bots in the largest connected piece, if that piece has no floppy mode (else 0)."""
    n = len(pos)
    roots = components(n, bonds)
    big = np.bincount(roots).argmax()
    keep = np.where(roots == big)[0]
    if len(keep) < 2:
        return 0.0
    sub = {int(k): a for a, k in enumerate(keep)}
    sb = [(sub[int(i)], sub[int(j)]) for i, j in bonds if int(i) in sub and int(j) in sub]
    f, _ = floppy_modes(np.asarray(pos)[keep], sb, **kw)
    return float(len(keep) / n) if f == 0 else 0.0


def bridges(n, bonds):
    """Bonds whose removal disconnects the graph (iterative DFS low-link)."""
    adj = [[] for _ in range(n)]
    for k, (i, j) in enumerate(bonds):
        adj[int(i)].append((int(j), k)); adj[int(j)].append((int(i), k))
    disc = [-1] * n; low = [0] * n; out = []; t = 0
    for s in range(n):
        if disc[s] != -1:
            continue
        stack = [(s, -1, iter(adj[s]))]; disc[s] = low[s] = t; t += 1
        while stack:
            v, pe, it = stack[-1]
            for w, k in it:
                if k == pe:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = t; t += 1
                    stack.append((w, k, iter(adj[w])))
                    break
                low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        out.append(pe)
    return out


def critical_fraction(n, bonds):
    """Fraction of bonds that are bridges (the cheap, connectivity-only version of 'critical')."""
    return len(bridges(n, bonds)) / max(len(bonds), 1)


def mean_degree(n, bonds):
    return 2.0 * len(bonds) / n
