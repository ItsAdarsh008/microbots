"""Turn the ideal dot into a physical microbot: wrap ANY cell without touching it.

    cell = embody.Embodied(cells.Cell(), mode="phys")          # or cells.build(embody={"mode": "phys"})

Same interface as the wrapped cell: mem_dims, in_dims, out_dims, init, param_count, and
step(params, x, s, mem, update, eps, eps0, block, total, mass) -> (x, s, mem, move). Anything else is
looked up on the wrapped cell. The decisions behind the numbers are in lab/embodiment.md (D3-D5).
All lengths are WORLD units (1 unit = 900 mm); body_r and port_d are read from
assets/microbot_v1/report.json when it exists (15 mm and 14 mm otherwise).

Modes
  ideal    nothing changes: step() is base.step() exactly.
  phys     sensing model + speed limit + no overlap (+ optional one-tick actuation latency).
  lattice  sensing model + bots live on a square docking lattice and hop one site per tick.

Sensing model (phys and lattice). The base cell is called on a *perceived* copy of the world:
  (a) positions + Gaussian noise (sense_noise, per particle, redrawn every tick). The noise is only in the copy
      the cell perceives; the proposed displacement (xout - xnoisy) is added to the TRUE position.
  (b) states quantised to quant_bits over [-1.5, 1.5] (clip, round; straight-through gradient). The cell's own
      state channel is quantised too (small simplification: a real bot knows its own state exactly). The true
      state keeps full precision: s_true += (s_out - s_quantised).
  (c) per-tick broadcast dropout: each particle's mass is multiplied by a Bernoulli(1 - dropout) draw, so
      neighbours do not see it that tick. A dropped particle still sees itself: while the cell runs,
      npa.perceive is temporarily replaced (trace time only, restored in a finally block) by a wrapper
      that adds the missing self term of the smoothed state, w(0) / (rho + w(0)) * s_i. Cells that use only
      neighbour lists (knn) exclude self anyway. Deleted particles (mass 0) are untouched in all modes.

Randomness. step() has no RNG argument, so the key is derived from the inputs: four scalars (random-weighted
sums of x, s, `update` and mass) are bit-cast to uint32 and folded into PRNGKey(seed). `update` is the
evaluator's fresh Bernoulli mask, so the key (hence the noise, dropout, tie-breaks) changes every tick and,
under vmap, per swarm; the same inputs always give the same outputs (deterministic, no hidden state). Limits: a
swarm that does not move, changes no state and gets an identical update mask twice sees identical randomness;
sums are float32 so it is a hash, not a cryptographic key. Embodied.step also accepts an optional trailing
key= keyword (not passed by run/evaluate) that overrides the derivation.

Bearing noise (optional, bearing_noise = sigma in rad). A real range-and-bearing sensor rotates each *neighbour
offset* by an independent random angle per (receiver, neighbour) pair. A black-box base.step only receives one
array of positions, so a pair-wise error is not implementable without re-writing every cell's neighbour code.
APPROXIMATION: each particle's perceived position gets extra isotropic Gaussian noise of std
bearing_noise * eps / 2 (the tangential error of a bearing error at half the sensing radius). It differs from the
real thing in that the error is shared by all receivers of that particle's position (correlated), so treat it as a
robustness knob, not a model of the sensor.

phys. A proposed displacement d goes through, in order: the deadzone (|d| < min_step -> 0; forward value 0 but the
gradient still flows, straight-through), the stride execution noise (length scaled by 1 + step_len_noise * N(0,1),
direction rotated by step_dir_noise * N(0,1) rad, drawn per particle per tick; applies to the executed move, so
with latency it is drawn when the move is proposed), and then the smooth cap:
  d -> d * M tanh(|d|/M) / |d|,  M = max_step: identity for small moves, gradient everywhere, exactly <= M.
Then proj_iters (3) rounds of Jacobi position-based projection: every live pair closer than 2 body_r is pushed
apart by half the overlap each along their line (the pushes on one particle are summed and scaled by `relax`).
Honest limits: the push is NOT speed-limited (a bot squeezed in a dense blob is displaced by more than max_step),
and 3 Jacobi rounds do not fully resolve a packing that starts far over-dense (the evaluator's seed disc at
eps 0.066 is ~16x over-dense): a residual overlap remains and is relaxed over the following ticks.
latency=1 delays the actuation by one tick: a move proposed at tick t is executed at t+1; the pending move is
stored in two extra mem channels, so mem_dims = base.mem_dims + 2 (the base cell never sees them).

lattice. Square lattice, pitch (default 2 * port_d = 28 mm), sites at integer multiples of pitch. Per tick a
selected particle with |d| > hop_threshold * pitch asks to hop to the neighbour site best aligned with d. A hop
is granted when the target is empty at the start of the tick (so sites being vacated this tick are not
usable), it wins the random tie against other claimants of the same site, and docking holds: every particle
that started with at least one lattice neighbour still has one after all granted hops (this covers both the
hopper and the neighbours it leaves; violating hops are cancelled in `dock_rounds` (4) sweeps, and if a
violation survives, all hops of that swarm are cancelled that tick: always safe).
hop_success < 1 makes each proposed hop fail with probability 1 - hop_success (the bot stays; it claims nothing).
A hopper that began isolated is exempt for itself. Positions of live particles are exactly ij * pitch.
Gradient: straight-through (forward = the hop, backward = x += d).
Off-lattice input (the evaluator's seed is a continuous disc, and the drag / noise / clump damages move
particles off the lattice) is snapped by snap_to_lattice() whenever any live particle is off by more than 1e-4
pitch: particles already on a site keep it, the rest go to the nearest free site (deterministic, greedy in
index order). A continuous seed disc that is 16x over-dense expands to a compact block on that first tick.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

import npa

MM = 1.0 / 900.0                    # world units per millimetre
MODES = ("ideal", "phys", "lattice")
LO, HI = -1.5, 1.5                   # range the state is quantised over


def _report_mm():
    path = Path(__file__).resolve().parents[3] / "assets" / "microbot_v1" / "report.json"
    try:
        p = json.loads(path.read_text())["params"]
        return float(p["body_r"]), float(p["port_d"])
    except Exception:                # file missing or different layout
        return 15.0, 14.0


BODY_R_MM, PORT_D_MM = _report_mm()

DEFAULTS = {
    "mode": "phys",
    "sense_noise": 3.0 * MM,         # position noise sigma (D3)
    "quant_bits": 8,                 # 0 / None: no quantisation
    "dropout": 0.1,                  # per-tick broadcast loss
    "max_step": 15.0 * MM,           # D4
    "body_r": BODY_R_MM * MM,
    "port_d": PORT_D_MM * MM,
    "pitch": None,                   # lattice pitch (default 2 * port_d, D5)
    "hop_threshold": 0.25,           # in pitches
    "proj_iters": 3,
    "relax": 1.0,
    "min_step": 0.0,                 # stride deadzone: a proposed move shorter than this becomes 0 (recommended 18 mm)
    "step_len_noise": 0.0,           # relative sigma of the executed stride length (recommended 0.2)
    "step_dir_noise": 0.0,           # sigma (rad) of the executed stride direction (recommended 0.087 = 5 deg)
    "bearing_noise": 0.0,            # sigma (rad) of the bearing error of a sensed neighbour (approximated, see docstring)
    "hop_success": 1.0,              # lattice: probability a proposed hop succeeds (recommended 0.9)
    "latency": 0,
    "dock_rounds": 4,
    "seed": 0,
}


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def quantise(s, bits):
    """Clip to [-1.5, 1.5] and round to 2^bits - 1 steps; straight-through gradient. Error <= half a step in range."""
    if not bits:
        return s
    step = (HI - LO) / (2**bits - 1)
    q = jnp.round((jnp.clip(s, LO, HI) - LO) / step) * step + LO
    return s + lax.stop_gradient(q - s)


def cap_move(d, max_step):
    """Smooth speed limit: d * M tanh(|d|/M) / |d|. Differentiable, |result| <= M, ~identity for |d| << M."""
    n = jnp.sqrt(jnp.sum(d * d, axis=-1, keepdims=True) + 1e-24)
    return d * (max_step * jnp.tanh(n / max_step) / n)


def project_overlaps(x, live, min_dist, iters, relax=1.0):
    """Jacobi position-based projection: live pairs closer than min_dist are pushed apart equally along their line."""
    n = x.shape[0]
    pair = (live[:, None] * live[None, :]) * (1.0 - jnp.eye(n, dtype=x.dtype))
    sgn = jnp.where(jnp.arange(n)[:, None] < jnp.arange(n)[None, :], 1.0, -1.0)      # direction for coincident points
    for _ in range(iters):
        diff = x[:, None, :] - x[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        diff = jnp.where((d2 < 1e-12)[..., None], sgn[..., None] * jnp.array([1e-3, 0.0], x.dtype), diff)
        d2 = jnp.sum(diff * diff, axis=-1)
        dist = jnp.sqrt(d2 + 1e-24)
        push = 0.5 * jnp.maximum(min_dist - dist, 0.0) * pair / dist
        x = x + relax * jnp.sum(push[..., None] * diff, axis=1)
    return x


def snap_to_lattice(x, mass=None, pitch=28.0 * MM, tol=1e-4):
    """Place live particles on distinct lattice sites (multiples of pitch): the nearest free site, deterministic.

    Particles already on a site (within tol * pitch) are placed first, so an on-lattice swarm is returned unchanged;
    the others follow in index order and take the nearest free site to their position. Deleted particles (mass 0)
    keep their position and occupy nothing. Search window: 3 sqrt(N) + 8 sites wide around the centroid.
    """
    n = x.shape[0]
    live = jnp.ones(n, bool) if mass is None else mass > 0
    xs = x / pitch
    ij0 = jnp.round(xs)
    onlat = jnp.max(jnp.abs(xs - ij0), axis=-1) <= tol
    order = jnp.argsort(jnp.where(live, jnp.where(onlat, 0, 1), 2), stable=True)
    g = int(3 * (n**0.5) + 8)
    centre = jnp.round(jnp.sum(jnp.where(live[:, None], xs, 0.0), axis=0) / jnp.maximum(jnp.sum(live), 1))
    origin = centre - g // 2
    gx = origin[0] + jnp.arange(g)
    gy = origin[1] + jnp.arange(g)

    def place(k, carry):
        occ, out = carry
        i = order[k]
        d2 = (gx - xs[i, 0])[:, None] ** 2 + (gy - xs[i, 1])[None, :] ** 2
        flat = jnp.argmin(jnp.where(occ, jnp.inf, d2))
        a, b = flat // g, flat % g
        occ = occ.at[a, b].set(occ[a, b] | live[i])
        site = jnp.stack([gx[a], gy[b]])
        out = out.at[i].set(jnp.where(live[i], site, xs[i]))
        return occ, out

    _, out = lax.fori_loop(0, n, place, (jnp.zeros((g, g), bool), xs))
    return jnp.where(live[:, None], out * pitch, x)


@contextmanager
def _patched_perceive(true_mass):
    """While active, npa.perceive gives a particle with effective mass 0 but true mass 1 (a dropped broadcast) its
    own smoothed-state term back: smooth_i += w(0) / (rho_i + w(0)) * s_i."""
    orig = npa.perceive

    def perceive(x, s, eps, eps0, block=None, total=None, mass=None):
        z = orig(x, s, eps, eps0, block, total, mass)
        n = x.shape[0]
        m = jnp.ones(n, x.dtype) if mass is None else mass
        xx = lax.stop_gradient(x)
        bs = n if block is None else block
        rho = lax.map(lambda xi: jnp.sum(npa._poly6(jnp.sum((xx[None] - xi[:, None]) ** 2, -1), eps) * m[None, :], -1),
                      xx.reshape(n // bs, bs, 2)).reshape(n)
        w0 = npa._poly6(jnp.zeros((), x.dtype), eps)
        need = (true_mass > 0) & (m <= 0)
        lo, hi = npa.STATE_DIMS, 2 * npa.STATE_DIMS
        extra = jnp.where(need[:, None], w0 / (rho + w0)[:, None] * s, 0.0)
        return z.at[:, lo:hi].add(extra)

    npa.perceive = perceive
    try:
        yield
    finally:
        npa.perceive = orig


def _bits(v):
    return lax.bitcast_convert_type(lax.stop_gradient(v).astype(jnp.float32), jnp.uint32)


def derive_key(seed, x, s, update, mass):
    """Deterministic per-tick key from the inputs (see module docstring)."""
    n = x.shape[0]
    i = jnp.arange(n, dtype=jnp.float32)
    w1, w2 = jnp.cos(1.0 + 0.7548776662 * i), jnp.sin(2.0 + 0.5698402910 * i)
    c = jnp.cos(0.3 + 1.3247179572 * jnp.arange(s.shape[-1], dtype=jnp.float32))
    scalars = (jnp.sum(x[:, 0] * w1 + x[:, 1] * w2),
               jnp.sum(s * c[None, :] * w1[:, None]),
               jnp.sum(update.reshape(n, -1)[:, 0] * w2 * (1.0 + i)),
               jnp.sum(mass * w1))
    key = jax.random.PRNGKey(seed)
    for v in scalars:
        key = jax.random.fold_in(key, _bits(v))
    return key


# ---------------------------------------------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------------------------------------------
class Embodied:
    def __init__(self, base, **spec):
        unknown = set(spec) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown embody options {sorted(unknown)}")
        self.base = base
        self.spec = {**DEFAULTS, **spec}
        c = self.spec
        if c["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if c["latency"] not in (0, 1):
            raise ValueError("latency must be 0 or 1")
        if c["mode"] == "lattice" and c["latency"]:
            raise ValueError("latency is only implemented for mode 'phys'")
        self.mode = c["mode"]
        self.pitch = float(c["pitch"] or 2.0 * c["port_d"])
        self._pend = 2 if (self.mode == "phys" and c["latency"]) else 0

    # ---- the cell interface -------------------------------------------------------------------------------------
    @property
    def mem_dims(self):
        return self.base.mem_dims + self._pend

    def __getattr__(self, name):             # in_dims, out_dims, cfg, scales, apply, ... come from the wrapped cell
        if name in ("base", "spec"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def init(self, key):
        return self.base.init(key)

    def param_count(self, params):
        return self.base.param_count(params)

    def snap_to_lattice(self, x, mass=None):
        return snap_to_lattice(x, mass, self.pitch)

    # ---- one tick -----------------------------------------------------------------------------------------------
    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None, key=None):
        c = self.spec
        if self.mode == "ideal":
            return self.base.step(p, x, s, mem, update, eps, eps0, block, total, mass)
        n = x.shape[0]
        live_m = jnp.ones(n, x.dtype) if mass is None else mass
        live = live_m > 0
        if key is None:
            key = derive_key(c["seed"], x, s, update, live_m)
        k_noise, k_drop, k_tie, k_stride, k_hop = jax.random.split(key, 5)
        bm = self.base.mem_dims
        bmem = mem[:, :bm]

        lattice = self.mode == "lattice"
        if lattice:                          # make sure everybody is on a site before anything is sensed
            xs = x / self.pitch
            off = jnp.max(jnp.where(live[:, None], jnp.abs(xs - jnp.round(xs)), 0.0))
            x_sn = lax.cond(off > 1e-4, lambda: snap_to_lattice(x, live_m, self.pitch), lambda: x)
            x = lax.stop_gradient(x_sn) + (x - lax.stop_gradient(x))

        # what the cell perceives
        sigma = c["sense_noise"]
        if c["bearing_noise"]:
            sigma = jnp.sqrt(sigma**2 + (c["bearing_noise"] * eps / 2.0) ** 2)
        xn = x + sigma * jax.random.normal(k_noise, x.shape, x.dtype)
        sq = quantise(s, c["quant_bits"])
        keep = jax.random.bernoulli(k_drop, 1.0 - c["dropout"], (n,)).astype(x.dtype)
        m_eff = live_m * keep
        if c["dropout"] > 0.0:
            with _patched_perceive(live_m):
                xo, so, bmem_o, move = self.base.step(p, xn, sq, bmem, update, eps, eps0, block, total, m_eff)
        else:
            xo, so, bmem_o, move = self.base.step(p, xn, sq, bmem, update, eps, eps0, block, total, mass)

        lv = live[:, None]
        d = jnp.where(lv, xo - xn, 0.0)      # what the cell asked for (0 where not updated / deleted)
        s_new = jnp.where(lv, s + (so - sq), s)
        bmem_o = jnp.where(lv, bmem_o, bmem)

        if lattice:
            return self._hop(x, d, update, live, k_tie, k_hop), s_new, bmem_o, move

        d = self._stride(d, k_stride)
        if self._pend:
            pend = mem[:, bm:] * lv
            mem_out = jnp.concatenate([bmem_o, d], axis=-1)
            d = pend
        else:
            mem_out = bmem_o
        x_new = project_overlaps(x + d, live_m, 2.0 * c["body_r"], c["proj_iters"], c["relax"])
        return jnp.where(lv, x_new, x), s_new, mem_out, move

    # ---- stride (phys) ------------------------------------------------------------------------------------------
    def _stride(self, d, key):
        """Deadzone, execution noise, then the smooth speed cap (so the cap bounds the executed move)."""
        c = self.spec
        if c["min_step"] > 0.0:
            ln = jnp.sqrt(jnp.sum(d * d, axis=-1, keepdims=True) + 1e-24)
            go = lax.stop_gradient(ln >= c["min_step"]).astype(d.dtype)
            d = go * d + (1.0 - go) * (d - lax.stop_gradient(d))          # below the deadzone: forward 0, backward identity
        if c["step_len_noise"] or c["step_dir_noise"]:
            kl, kd = jax.random.split(key)
            scale = jnp.maximum(1.0 + c["step_len_noise"] * jax.random.normal(kl, d.shape[:1] + (1,), d.dtype), 0.0)
            ang = c["step_dir_noise"] * jax.random.normal(kd, d.shape[:1], d.dtype)
            co, si = jnp.cos(ang), jnp.sin(ang)
            d = scale * jnp.stack([co * d[:, 0] - si * d[:, 1], si * d[:, 0] + co * d[:, 1]], axis=-1)
        return cap_move(d, c["max_step"])

    # ---- lattice hop --------------------------------------------------------------------------------------------
    def _hop(self, x, d, update, live, k_tie, k_hop):
        c, pitch = self.spec, self.pitch
        n = x.shape[0]
        ij = jnp.round(lax.stop_gradient(x) / pitch).astype(jnp.int32)
        dirs = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]], jnp.int32)
        sd = lax.stop_gradient(d)
        pick = jnp.argmax(jnp.stack([sd[:, 0], -sd[:, 0], sd[:, 1], -sd[:, 1]], axis=-1), axis=-1)
        target = ij + dirs[pick]
        want = live & (update[:, 0] > 0.5) & (jnp.sqrt(jnp.sum(sd * sd, -1)) > c["hop_threshold"] * pitch)
        if c["hop_success"] < 1.0:                                   # teeth slip: the hop fails and the bot stays
            want = want & jax.random.bernoulli(k_hop, c["hop_success"], (n,))

        def same(a, b):                       # (N, N): site of a_i == site of b_j
            return jnp.all(a[:, None, :] == b[None, :, :], axis=-1)

        def adjacent(sites):                  # (N, N): live pair on neighbouring sites
            l1 = jnp.sum(jnp.abs(sites[:, None, :] - sites[None, :, :]), axis=-1)
            return (l1 == 1) & live[:, None] & live[None, :]

        eye = jnp.eye(n, dtype=bool)
        blocked = jnp.any(same(target, ij) & live[None, :], axis=1)
        claim = want & ~blocked
        pri = jax.random.uniform(k_tie, (n,))
        idx = jnp.arange(n)
        earlier = (pri[None, :] < pri[:, None]) | ((pri[None, :] == pri[:, None]) & (idx[None, :] < idx[:, None]))
        lost = jnp.any(same(target, target) & claim[None, :] & earlier & ~eye, axis=1)
        grant = claim & ~lost

        adj0 = adjacent(ij)
        docked0 = jnp.any(adj0, axis=1)
        for _ in range(c["dock_rounds"]):
            bad = docked0 & ~jnp.any(adjacent(jnp.where(grant[:, None], target, ij)), axis=1)
            left_bad = jnp.any(adj0 & bad[:, None], axis=0)          # hoppers that were next to a now-isolated particle
            grant = grant & ~bad & ~left_bad
        bad = docked0 & ~jnp.any(adjacent(jnp.where(grant[:, None], target, ij)), axis=1)
        grant = grant & ~jnp.any(bad)                                # last resort: nobody hops this tick

        site = jnp.where(grant[:, None], target, ij)
        pos = site.astype(x.dtype) * pitch                           # exact site coordinates
        xd = x + d
        out = pos + (xd - lax.stop_gradient(xd))                     # forward: the site; backward: x += d
        return jnp.where(live[:, None], out, x)
