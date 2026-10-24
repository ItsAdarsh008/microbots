"""An ENGINEERED (nothing learned) per-particle rule: "rest memory". A ceiling for what a purely local rule that
remembers its neighbours' geometry can do on task 2b (shape-agnostic healing), to compare learned cells against.

    {"kind": "rest", "K": 6, "imprint_at": 16, "gain": 0.5, "colour_rate": 0.3, "sense_r": 1.0, "max_move": 0.5,
     "frame": "world" | "distance", "diffuse_k": 12, "forget": False}

Private memory per particle (mem_dims = 4 + 3 K, a float vector that neighbours never read, except for the age, see below):
    [ age (1) | own colour (3, stored as s[-3:]) | K neighbour indices as floats (-1 = empty; exact below 2^24) | 2K offsets ]
age counts the particle's OWN updates (the `update` mask), capped at imprint_at + 1. All-zero memory (age 0) is "amnesiac",
which is what wipe damage produces (and what the evaluator's start swarm has).

Phase 1 (age < imprint_at, amnesiacs included): the particle does not move. Its colour channels (last 3 of s) relax at
colour_rate toward the mean colour of the currently sensed live neighbours (within eps; the diffuse_k nearest, via
neighbors.knn) that are themselves in phase 2 (age >= imprint_at; a neighbour could broadcast one bit for this). With no such
neighbour nothing happens.
Imprint (the one update at which age goes imprint_at - 1 -> imprint_at): store the indices of the K nearest live neighbours
within eps (padded with -1), their current offsets x_j - x_i, and the own colour (after that update's phase-1 colour step).
Phase 2 (age >= imprint_at): for each remembered neighbour j that is alive and currently within sense_r * eps, the residual
r_ij = (x_j - x_i) - o_ij; the particle moves dx = gain * mean_j r_ij (0 if no valid neighbour), clipped to max_move * eps,
and relaxes its colour toward the remembered one: s[-3:] += colour_rate * (c_own - s[-3:]). Other channels: ds = 0.
Moves, state and memory are applied only for particles with update = 1 and mass > 0, as in cells.Cell.step.

Locality / honesty. A particle uses only: live neighbours within eps and their age (at imprint and in phase 1); afterwards the
remembered ids and offsets and the CURRENT positions of remembered neighbours within sense_r * eps (a farther one is not
used, because a bot could not sense it). Nothing global.
LIMITATION (frame='world'): the remembered offsets are in the WORLD frame. Real bots have no shared compass and no heading, so
this variant assumes bots that can measure the bearing to a neighbour in a fixed frame; it also cannot cope with the swarm
being rotated. frame='distance' removes that: it remembers only the scalar distance |o_ij| (stored in the first K offset
slots) and moves along the CURRENT bearing to j by (|x_j - x_i| - |o_ij|) * gain, averaged over valid j. That rule is
rotation-invariant but only restores distances, so it can leave a mirrored/sheared patch (distances alone do not fix a shape).

Options beyond the specification, both documented here and reported with results:
    diffuse_k  how many nearest neighbours phase 1 looks at for phase-2 neighbours (default 12; with K = 6 nearest it can
               happen that none of them is in phase 2 although a farther one is)
    forget     (default False) a phase-2 particle that remembers neighbours but none of them is alive and within sense_r * eps
               drops back to amnesia (age 0) and re-imprints; this heals shuffled memory (wrong ids) at the price of
               forgetting at the edge of a cut. Off = the specified rule.
Known weakness (by design): neighbours-of-amnesiacs wait until imprint_at updates of their own before they can serve as a
source, so a large wiped patch is refilled in waves of about 2 * imprint_at ticks each, and during warm-up early imprinters
make still-amnesiac neighbours blur their colour toward the neighbourhood mean (the start swarm is indistinguishable from
a wiped one).
"""

import jax.numpy as jnp

import neighbors

DEFAULTS = {
    "K": 6,                 # remembered neighbours
    "imprint_at": 16,       # own updates before the geometry is memorised
    "gain": 0.5,
    "colour_rate": 0.3,
    "sense_r": 1.0,         # multiples of eps within which a remembered neighbour can currently be sensed
    "max_move": 0.5,        # per update, multiples of eps
    "frame": "world",       # world: remember offsets | distance: remember only |offset|, move along the current bearing
    "diffuse_k": 12,
    "forget": False,
}


class Cell:
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown cell options {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        if c["frame"] not in ("world", "distance"):
            raise ValueError(f"bad frame {c['frame']}")
        self.K = int(c["K"])
        self.mem_dims = 4 + 3 * self.K

    def init(self, key):
        return {}

    def param_count(self, params):
        return 0

    def step(self, params, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        c, K, n = self.cfg, self.K, x.shape[0]
        T = float(c["imprint_at"])
        mass = jnp.ones(n, x.dtype) if mass is None else mass
        live = (mass > 0).astype(x.dtype)
        act = update[:, 0] * live                                     # (N,) who acts
        age, own, ids, off = mem[:, 0], mem[:, 1:4], mem[:, 4:4 + K], mem[:, 4 + K:].reshape(n, K, 2)
        phase2 = age >= T - 0.5

        # ---- what is remembered and still sensed: remembered neighbours alive and within sense_r * eps
        j = jnp.clip(ids.astype(jnp.int32), 0, n - 1)
        dvec = x[j] - x[:, None, :]                                    # (N, K, 2) current offsets to the remembered ids
        dist = jnp.sqrt(jnp.sum(dvec * dvec, axis=-1) + 1e-24)
        ok = (ids > -0.5) & (mass[j] > 0) & (dist < c["sense_r"] * eps)
        okf = ok.astype(x.dtype)
        nok = jnp.sum(okf, axis=-1)
        if c["frame"] == "world":
            resid = dvec - off
        else:
            bearing = dvec / dist[..., None]
            resid = bearing * (dist - off[..., 0])[..., None]
        dx = c["gain"] * jnp.sum(resid * okf[..., None], axis=1) / jnp.maximum(nok, 1.0)[:, None]
        length = jnp.sqrt(jnp.sum(dx * dx, axis=-1, keepdims=True) + 1e-24)
        dx = dx * jnp.minimum(1.0, c["max_move"] * eps / length)

        if c["forget"]:
            lost = phase2 & jnp.any(ids > -0.5, axis=-1) & (nok < 0.5)
            phase2 = phase2 & ~lost
        p2 = phase2.astype(x.dtype)
        dx = dx * p2[:, None]                                          # phase 1 does not move

        # ---- phase 1: colour diffusion from phase-2 neighbours, and the neighbour lists for the imprint
        kk = max(K, int(c["diffuse_k"]))
        nidx, nvalid, noff = neighbors.knn(x, mass, eps, kk)
        nage = mem[nidx, 0]
        use = nvalid * (nage >= T - 0.5).astype(x.dtype)
        cnt = jnp.sum(use, axis=-1, keepdims=True)
        mean_col = jnp.sum(use[..., None] * s[nidx][..., -3:], axis=1) / jnp.maximum(cnt, 1.0)
        d_col1 = jnp.where(cnt > 0, c["colour_rate"] * (mean_col - s[:, -3:]), 0.0)
        d_col2 = c["colour_rate"] * (own - s[:, -3:])
        d_col = jnp.where(phase2[:, None], d_col2, d_col1)
        ds = jnp.concatenate([jnp.zeros((n, s.shape[1] - 3), s.dtype), d_col], axis=-1)

        # ---- memory update
        imprint = (~phase2) & (age >= T - 1.5) & (age < T - 0.5) & (act > 0)
        nn_ids = jnp.where(nvalid[:, :K] > 0, nidx[:, :K].astype(x.dtype), -1.0)
        nn_off = noff[:, :K] * nvalid[:, :K, None]
        if c["frame"] == "distance":
            dd = jnp.sqrt(jnp.sum(nn_off * nn_off, axis=-1))
            nn_off = jnp.stack([dd, jnp.zeros_like(dd)], axis=-1)
        new_own = s[:, -3:] + d_col
        i_col = imprint[:, None]
        mem_ids = jnp.where(i_col, nn_ids, ids)
        mem_off = jnp.where(i_col[..., None], nn_off, off)
        mem_own = jnp.where(i_col, new_own, own)
        new_age = jnp.minimum(jnp.where(phase2, age, jnp.where(age >= T - 0.5, 0.0, age)) + 1.0, T + 1.0)
        new_mem = jnp.concatenate([new_age[:, None], mem_own, mem_ids, mem_off.reshape(n, 2 * K)], axis=-1)
        mem = mem + update * live[:, None] * (new_mem - mem)

        move = jnp.mean(jnp.sqrt(jnp.sum(dx * dx, axis=-1) + 1e-24))
        a = act[:, None]
        return x + a * dx, s + a * ds, mem, move
