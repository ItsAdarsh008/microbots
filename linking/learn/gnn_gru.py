"""A graph neural network with GRU memory, as the bots' shared policy.

The goal: with many more bots than the shape needs (3x, 5x),

  (a) only the bots actually needed should go to work; the rest stay put, and
  (b) once the shape is finished, the spares should go home to the pad where
      they started, not sit in a heap on top of the structure.

Both depend on things no single bot can see: whether it's one of the bots
nearest the work, and whether the shape is finished somewhere out of view.
The hand-written rules (the teacher, env.plan) do both by looking at the
whole grid. The network has to work them out locally:

    * GNN: every step, each bot sends a message to the bots touching it (the 26
      cells around it) and combines what it hears, 3 rounds per step. That's
      how news moves through the swarm.
    * GRU: each bot keeps a hidden state from step to step, updated by a GRU
      cell. That's memory, so news can keep travelling over many steps ("the
      work is that way", "everything's built, go home").

Each bot sees its own 5x5x5 block (which cells hold a bot, which belong to the
shape), whether it's locked in or parked, and the relayed "how far to the
work" hop count that bots pass to each other (pointing home once the shape is
built). No compass (USE_COMPASS turns it on). Every bot runs the same network.

Training is imitation: record the teacher, train the network to pick the
teacher's move, then DAgger (the network drives, the teacher labels).

    python linking/learn/gnn_gru.py --quick              # small smoke test (a few minutes on a CPU)
    python linking/learn/gnn_gru.py                      # the real run: meant for a GPU (Colab)
    python linking/learn/gnn_gru.py --no-memory          # same, but the GRU state is wiped every step

The data recording and the evaluation run the Python simulation, on the CPU.
The training is the heavy part, and a GPU makes it many times faster.
"""

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import env as E

HERE = os.path.dirname(os.path.abspath(__file__))
L = E.L

N_FEAT = 2 * E.SIDE ** 3 + 2 + 7 + 4  # 125 occupied + 125 shape + locked + parked + 7 relayed signal + 4 compass = 263
USE_SIGNAL = True                      # the relayed "how far to the work" hop count (local: bots pass it along)
USE_COMPASS = False                    # direction to the nearest open cell (needs live state from far away: a shortcut)
N_BYTES = N_FEAT                       # stored as bytes
NEIGHBOURS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                       if (a, b, c) != (0, 0, 0)])                # 26 cells touching a bot
WINDOW = 32                            # steps per training window (truncated backprop through time)
N_PAD = 576                            # bots per build, padded; set by configure()
E_PAD = N_PAD * len(NEIGHBOURS)


def configure(mult, shapes=("sheet", "arch", "tower")):
    """Size the padding for `mult` times the usual number of bots."""
    global N_PAD, E_PAD
    biggest = max((int(L.SHAPES[s]().sum()) + 12) * mult for s in shapes)
    N_PAD = 64 * int(np.ceil((biggest + 1) / 64))
    E_PAD = N_PAD * len(NEIGHBOURS)
    return N_PAD


THRIFTY, PARK = "rank", True           # the teacher's behaviours (see env.prepare); --easy turns both off


def new_swarm(shape, seed, mult):
    need = int(L.SHAPES[shape]().sum())
    sw = L.Swarm(L.SHAPES[shape](), seed=seed, extra=(need + 12) * mult - need)
    return E.prepare(sw, thrifty=THRIFTY, park=PARK)


# ============================================================== recording

def features(sw, ctx):
    """What each bot sees, as bytes: (N_PAD, N_FEAT)."""
    frame = E.frame_state(sw, ctx["goal"], ctx["dist"])
    x = np.zeros((N_PAD, N_FEAT), dtype=np.uint8)    # 0/1 cells, then signal and compass as 0..255
    for b in range(len(sw.pos)):
        p = tuple(int(v) for v in sw.pos[b])
        obs = E.observe(frame, p)
        x[b, :250] = obs[:250]
        x[b, 250] = obs[257]                       # locked into the shape
        x[b, 251] = sw.home[p]                     # parked on the pad
        if USE_SIGNAL:                             # how far to the work (home, once built), at the bot and its 6 faces
            x[b, 252:259] = np.round(obs[250:257] * 255)
        if USE_COMPASS:
            x[b, 259:263] = np.round((obs[258:262] + 1) * 127.5)
    return x


def edges(sw):
    """Who can hear whom this step: every pair of bots in touching cells. (senders, receivers, offsets)."""
    pos = np.asarray(sw.pos)
    n = len(pos)
    snd, rcv, off = [], [], []
    for d in NEIGHBOURS:
        q = pos + d
        ok = np.all((q >= 0) & (q < np.array(L.GRID)), axis=1)
        idx = np.full(n, -1)
        idx[ok] = sw.id_at[tuple(q[ok].T)]
        has = idx >= 0
        snd.append(idx[has]); rcv.append(np.nonzero(has)[0]); off.append(np.repeat(d[None], has.sum(), 0))
    return (np.concatenate(snd).astype(np.int16), np.concatenate(rcv).astype(np.int16),
            np.concatenate(off).astype(np.int8))


def teacher_labels(sw, ctx):
    """For every bot that may move: its legal moves, and what the teacher would do."""
    busy = sw.occ.copy()
    mask = np.zeros((N_PAD, E.N_ACTIONS), dtype=bool)
    mask[:, 0] = True
    label = np.zeros(N_PAD, dtype=np.int8)
    train = np.zeros(N_PAD, dtype=bool)
    need = np.zeros(N_PAD, dtype=bool)
    for p in map(tuple, np.argwhere(ctx["movers"])):
        b = sw.id_at[p]
        slots = E.legal_slots(sw, p, busy)
        for s in slots:
            mask[b, s] = True
        order = E.rule_chooser(sw, p, slots, ctx)
        label[b] = order[0] if order else 0
        train[b] = True
        need[b] = ctx["needed"] is None or bool(ctx["needed"][p])
    return mask, label, train, need


def rollout(shape, seed, mult, driver=None, build_steps=200, park_steps=150):
    """One build and the spares going home, recording every step. driver=None: the teacher drives."""
    sw = new_swarm(shape, seed, mult)
    assert len(sw.pos) < N_PAD, "configure() for this many bots first"
    rec = {k: [] for k in ("x", "snd", "rcv", "off", "mask", "label", "train", "need")}
    if driver is not None:
        driver.reset()
    t = parked_for = 0
    built_at = None
    while not E.all_home(sw):
        ctx = E.plan(sw)
        if ctx is None:                             # built, and no trip home in this mode
            built_at = t if built_at is None else built_at
            break
        if sw.done():
            built_at = t if built_at is None else built_at
            if parked_for >= park_steps:
                break
            parked_for += 1
        elif t >= build_steps:
            break
        x = features(sw, ctx)
        snd, rcv, off = edges(sw)
        mask, label, train, need = teacher_labels(sw, ctx)
        for k, v in zip(rec, (x, snd, rcv, off, mask, label, train, need)):
            rec[k].append(v)
        if driver is None:
            E.step(sw, E.rule_chooser)
        else:
            driver.step(sw, x, snd, rcv, off)
        t += 1
    return rec, stats(sw, built_at, parked_for)


def stats(sw, built_at, parked_for):
    spares = [b for b in range(len(sw.pos)) if not sw.locked[tuple(sw.pos[b])]]
    moved = len(set(spares) & getattr(sw, "moved_ever", set()))
    return {"bots": len(sw.pos), "spares": len(spares), "built_in": built_at,
            "spares_that_moved": moved,
            "home_in": parked_for if E.all_home(sw) else None,
            "not_home": int((sw.occ & ~sw.locked & ~sw.home).sum()),
            "attached": bool(L.is_connected(sw.occ))}


# ============================================================== the network

H = 64          # hidden size
ROUNDS = 3      # message-passing rounds per step


def dense(key, n_in, n_out):
    return {"w": jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in), "b": jnp.zeros(n_out)}


def init(key):
    ks = iter(jax.random.split(key, 32))
    return {
        "enc1": dense(next(ks), N_FEAT, H), "enc2": dense(next(ks), H, H),
        "from_mem": dense(next(ks), H, H),
        "msg": [dense(next(ks), 2 * H + 3, H) for _ in range(ROUNDS)],
        "upd": [dense(next(ks), 2 * H, H) for _ in range(ROUNDS)],
        "head1": dense(next(ks), 2 * H, H), "head2": dense(next(ks), H, E.N_ACTIONS),
        "go": dense(next(ks), H, 1),          # should this bot work right now? (one number)
        # GRU cell: update gate z, reset gate r, candidate n
        "gru_x": dense(next(ks), H, 3 * H), "gru_h": dense(next(ks), H, 3 * H),
    }


def lin(p, x):
    return x @ p["w"] + p["b"]


def gru(p, h, x):
    """One GRU step. z: how much of the old memory to keep; r: how much of it to use for the new candidate."""
    zx, rx, nx = jnp.split(lin(p["gru_x"], x), 3, axis=-1)
    zh, rh, nh = jnp.split(lin(p["gru_h"], h), 3, axis=-1)
    z = jax.nn.sigmoid(zx + zh)
    r = jax.nn.sigmoid(rx + rh)
    n = jnp.tanh(nx + r * nh)
    return (1 - z) * n + z * h


def step_fn(p, h, xbits, snd, rcv, off, memory=True):
    """One simulation step for every bot: listen to neighbours, update memory, score the 19 moves."""
    x = xbits.astype(jnp.float32)
    x = x.at[..., 252:].multiply(1 / 255.0)          # signal and compass back to 0..1
    snd, rcv, off = snd.astype(jnp.int32), rcv.astype(jnp.int32), off.astype(jnp.float32)
    if not memory:
        h = jnp.zeros_like(h)
    u = jax.nn.relu(lin(p["enc2"], jax.nn.relu(lin(p["enc1"], x)))) + lin(p["from_mem"], h)
    deg = jax.ops.segment_sum(jnp.ones_like(rcv, dtype=jnp.float32), rcv, num_segments=N_PAD)
    deg = jnp.maximum(deg, 1.0)[:, None]
    for k in range(ROUNDS):
        msg = jax.nn.relu(lin(p["msg"][k], jnp.concatenate([u[snd], u[rcv], off], axis=-1)))
        agg = jax.ops.segment_sum(msg, rcv, num_segments=N_PAD) / deg          # mean over neighbours
        u = u + jax.nn.relu(lin(p["upd"][k], jnp.concatenate([u, agg], axis=-1)))
    h2 = gru(p, h, u)
    z = jax.nn.relu(lin(p["head1"], jnp.concatenate([u, h2], axis=-1)))
    return h2, lin(p["head2"], z), lin(p["go"], z)[:, 0]


def count(p):
    return sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(p))


# ============================================================== training

def window_loss(p, h0, w, memory, go_weight, stay_weight):
    """Loss over one window of steps for one build, carrying each bot's memory through it.

    Two parts, so they can't crowd each other out:
    * go: should this bot move right now? Matches whether the teacher moves it:
      not if it isn't needed, and not if it has no move that helps. Moving and
      staying are weighted so each counts about equally overall, whichever is rarer.
    * which move: only for bots the teacher actually moves, which of the
      legal moves (stay excluded).
    """
    def body(h, s):
        h, logits, go = step_fn(p, h, s["x"], s["snd"], s["rcv"], s["off"], memory)
        tr = s["train"]
        need = (s["label"] != 0) & tr                # the teacher moves this bot this step
        # go / stay
        wt = jnp.where(need, go_weight, stay_weight) * tr
        bce = jnp.where(need, jax.nn.softplus(-go), jax.nn.softplus(go))
        go_hit = (go > 0) == need
        # which move
        moves = s["mask"].at[:, 0].set(False)
        lm = jnp.where(moves, logits, -1e9)
        logp = jax.nn.log_softmax(lm)
        lab = s["label"].astype(jnp.int32)
        mv = need & (lab > 0)
        nll = -jnp.take_along_axis(logp, lab[:, None], axis=1)[:, 0]
        mv_hit = jnp.argmax(lm, axis=1) == lab
        return h, (jnp.sum(bce * wt), jnp.sum(wt), jnp.sum(nll * mv), jnp.sum(mv),
                   jnp.sum(go_hit * need), jnp.sum(need), jnp.sum(go_hit * tr * ~need), jnp.sum(tr * ~need),
                   jnp.sum(mv_hit * mv))
    h, o = jax.lax.scan(body, h0, w)
    o = [x.sum() for x in o]
    loss = o[0] / jnp.maximum(o[1], 1.0) + o[2] / jnp.maximum(o[3], 1.0)
    return loss, (h, jnp.stack([o[4], o[5], o[6], o[7], o[8], o[3]]))


def batched_loss(p, h0, wb, memory, go_weight, stay_weight):
    loss, (h, c) = jax.vmap(window_loss, in_axes=(None, 0, 0, None, None, None))(p, h0, wb, memory, go_weight, stay_weight)
    return loss.mean(), (h, c.sum(axis=0))


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": 0}


def make_train_step(memory):
    @jax.jit
    def train_step(p, opt, h0, wb, lr, go_weight, stay_weight):
        (loss, (h, counts)), g = jax.value_and_grad(batched_loss, has_aux=True)(p, h0, wb, memory, go_weight, stay_weight)
        g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), g)
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, opt["m"], g)
        v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, opt["v"], g)
        p = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, {"m": m, "v": v, "t": t}, jax.lax.stop_gradient(h), loss, counts
    return train_step


def window(builds, idx, t0):
    """Steps t0 .. t0+WINDOW of the builds in idx, padded to fixed sizes (padding trains nothing)."""
    B, W = len(idx), WINDOW
    out = {"x": np.zeros((B, W, N_PAD, N_BYTES), np.uint8),
           "snd": np.full((B, W, E_PAD), N_PAD - 1, np.int16), "rcv": np.full((B, W, E_PAD), N_PAD - 1, np.int16),
           "off": np.zeros((B, W, E_PAD, 3), np.int8),
           "mask": np.zeros((B, W, N_PAD, E.N_ACTIONS), bool), "label": np.zeros((B, W, N_PAD), np.int8),
           "train": np.zeros((B, W, N_PAD), bool), "need": np.zeros((B, W, N_PAD), bool)}
    out["mask"][..., 0] = True
    for i, b in enumerate(idx):
        rec = builds[b]
        for j in range(W):
            t = t0 + j
            if t >= len(rec["x"]):
                break
            m = len(rec["snd"][t])
            out["x"][i, j] = rec["x"][t]
            out["snd"][i, j, :m] = rec["snd"][t]; out["rcv"][i, j, :m] = rec["rcv"][t]; out["off"][i, j, :m] = rec["off"][t]
            for k in ("mask", "label", "train", "need"):
                out[k][i, j] = rec[k][t]
    return {k: jnp.asarray(v) for k, v in out.items()}


def train(p, opt, builds, epochs, memory, log, batch=4, lr=1e-3, stage="imitation"):
    batch = min(batch, len(builds))
    step = make_train_step(memory)
    rng = np.random.default_rng(len(log["loss"]))
    n_go = sum(int(((l != 0) & t).sum()) for b in builds for l, t in zip(b["label"], b["train"]))
    n_wait = sum(int(((l == 0) & t).sum()) for b in builds for l, t in zip(b["label"], b["train"]))
    n_all = max(n_go + n_wait, 1)
    go_weight = float(np.clip(n_all / (2 * max(n_go, 1)), 0.02, 50.0))
    stay_weight = float(np.clip(n_all / (2 * max(n_wait, 1)), 0.02, 50.0))
    print(f"  {n_go:,} decisions where the teacher moves the bot, {n_wait:,} where it stays "
          f"(weights: move {go_weight:.2f}, stay {stay_weight:.2f})")
    for ep in range(epochs):
        order = rng.permutation(len(builds))
        ls, tot = [], np.zeros(6)
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            T = max(len(builds[b]["x"]) for b in idx)
            h = jnp.zeros((batch, N_PAD, H))
            for t0 in range(0, T, WINDOW):
                p, opt, h, loss, counts = step(p, opt, h, window(builds, idx, t0), lr, go_weight, stay_weight)
                ls.append(float(loss)); tot += np.asarray(counts)
                log["loss"].append(round(float(loss), 4))
        go_ok, wait_ok, move_ok = tot[0] / max(tot[1], 1), tot[2] / max(tot[3], 1), tot[4] / max(tot[5], 1)
        log["epoch_end"].append(len(log["loss"])); log["agree"].append(round(float(move_ok), 4))
        log.setdefault("go_ok", []).append(round(float(go_ok), 4)); log.setdefault("wait_ok", []).append(round(float(wait_ok), 4))
        log["stage"].append(stage)
        print(f"  {stage:9s} epoch {len(log['agree']):3d}  loss {np.mean(ls):.3f}  |  moves when it should {go_ok:.0%}, "
              f"stays when it should {wait_ok:.0%}  |  same move as the teacher {move_ok:.0%}", flush=True)
    return p, opt


# ============================================================== driving the swarm

class Driver:
    """The trained network driving every bot, with a GRU state per bot."""

    def __init__(self, p, memory=True, seed=0, go_cut=None):
        # go_cut=None: move with the probability the network gives. A number: move only if
        # that probability is above it (no dice, so a bot told "probably stay" never drifts).
        self.p, self.memory, self.go_cut = p, memory, go_cut
        self.rng = np.random.default_rng(seed)
        self.fwd = jax.jit(lambda p, h, x, s, r, o: step_fn(p, h, x, s, r, o, memory))

    def reset(self):
        self.h = jnp.zeros((N_PAD, H))

    def step(self, sw, x, snd, rcv, off):
        m = len(snd)
        S = np.full(E_PAD, N_PAD - 1, np.int16); R = np.full(E_PAD, N_PAD - 1, np.int16)
        O = np.zeros((E_PAD, 3), np.int8)
        S[:m], R[:m], O[:m] = snd, rcv, off
        self.h, logits, go = self.fwd(self.p, self.h, jnp.asarray(x), jnp.asarray(S), jnp.asarray(R), jnp.asarray(O))
        logits, go = np.asarray(logits), np.asarray(go)

        def chooser(sw_, pcell, slots, ctx):
            b = sw_.id_at[pcell]
            # move with the probability the network gives (a hard cut-off makes whole
            # groups freeze together when they're all just under it)
            pgo = 1 / (1 + np.exp(-go[b]))
            if not slots or (pgo <= self.go_cut if self.go_cut is not None else self.rng.random() > pgo):
                return []                              # stay put this step
            options = list(slots)
            s = logits[b, options]
            prob = np.exp(s - s.max()); prob /= prob.sum()
            first = options[self.rng.choice(len(options), p=prob)]
            rest = [o for o in sorted(options, key=lambda o: -logits[b, o]) if o != first]
            return [first] + rest
        E.step(sw, chooser)


def evaluate(p, memory, mult, shapes=("sheet", "arch", "tower"), seeds=range(100, 105), who="network", go_cut=0.5):
    """Build each shape and send the spares home. who="teacher" runs the hand-written rules instead."""
    rows = []
    for shape in shapes:
        for seed in seeds:
            driver = None if who == "teacher" else Driver(p, memory, seed, go_cut)
            _, st = rollout(shape, seed, mult, driver)
            st.update(shape=shape, seed=seed, who=who)
            rows.append(st)
            print(f"  {who:8s} {shape:6s} seed {seed}: built in {st['built_in']}, "
                  f"{st['spares_that_moved']}/{st['spares']} spares moved, "
                  f"home in {st['home_in']} ({st['not_home']} not home)", flush=True)
    return rows


# ============================================================== main

def main():
    global USE_SIGNAL, USE_COMPASS, THRIFTY, PARK, ROUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--no-memory", action="store_true", help="wipe the GRU state every step (comparison)")
    ap.add_argument("--mult", type=int, default=3, help="times the usual number of bots")
    ap.add_argument("--seeds", type=int, default=12, help="teacher builds per training shape")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dagger", type=int, default=2, help="DAgger rounds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-only", action="store_true", help="load saved weights and just test them")
    ap.add_argument("--eval-seeds", type=int, default=3, help="unseen builds per shape to test on")
    ap.add_argument("--compass", action="store_true", help="also give each bot the compass (a shortcut)")
    ap.add_argument("--no-signal", action="store_true", help="don't give bots the relayed signal")
    ap.add_argument("--easy", action="store_true", help="plain building only: every bot works, no trip home (sanity check)")
    ap.add_argument("--rounds", type=int, default=ROUNDS, help="message-passing rounds per step (0 = no GNN, a plain MLP)")
    a = ap.parse_args()
    memory = not a.no_memory
    USE_SIGNAL, USE_COMPASS = not a.no_signal, a.compass
    if a.easy:
        THRIFTY, PARK = False, False
    ROUNDS = a.rounds
    if a.quick:
        a.mult, a.seeds, a.epochs, a.dagger = 2, 2, 3, 1
    configure(a.mult)
    name = ("gnn_gru" if memory else "gnn_nomem") + f"_x{a.mult}" + ("_compass" if a.compass else "") + ("_nosignal" if a.no_signal else "") + ("_easy" if a.easy else "") + (f"_r{a.rounds}" if a.rounds != 3 else "")
    out = a.out or os.path.join(HERE, name)
    shapes = ["sheet", "arch"]
    print("devices:", jax.devices(), f"| {a.mult}x bots, padded to {N_PAD}")

    def save(p, log):
        with open(out + ".pkl", "wb") as f:
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p), "memory": memory, "mult": a.mult}, f)
        json.dump(log, open(out + "_log.json", "w"))

    if a.eval_only:
        p = pickle.load(open(out + ".pkl", "rb"))["params"]
    else:
        # the recorded builds are cached, so a rerun (or a crash) doesn't have to record them again
        cache = out + "_teacher2.pkl"           # "2": recordings that include the teacher's needed flag
        if os.path.exists(cache):
            builds = pickle.load(open(cache, "rb"))
            print(f"loaded {len(builds)} recorded builds from {cache}")
        else:
            t0 = time.time()
            print("recording the teacher...")
            builds = []
            for shape in shapes:
                for seed in range(a.seeds):
                    rec, st = rollout(shape, seed, a.mult)
                    builds.append(rec)
                    print(f"  teacher {shape:5s} seed {seed:2d}: built in {st['built_in']}, {st['spares_that_moved']}/"
                          f"{st['spares']} spares moved, home in {st['home_in']}", flush=True)
            print(f"recorded {len(builds)} builds in {time.time() - t0:.0f}s")
            pickle.dump(builds, open(cache, "wb"))

        p = init(jax.random.PRNGKey(0))
        opt = adam_init(p)
        print(f"network: {count(p):,} weights, {ROUNDS} message rounds per step, GRU memory {'on' if memory else 'off'}")
        log = {"loss": [], "epoch_end": [], "agree": [], "stage": []}
        t0 = time.time()
        p, opt = train(p, opt, builds, a.epochs, memory, log)
        print(f"imitation done in {time.time() - t0:.0f}s")
        save(p, log)

    for r in range(0 if a.eval_only else a.dagger):
        print(f"DAgger round {r + 1}: the network drives, the teacher labels")
        for shape in shapes:
            for seed in range(1000 + 10 * r, 1000 + 10 * r + max(1, a.seeds // 2)):
                rec, st = rollout(shape, seed, a.mult, Driver(p, memory, seed))
                builds.append(rec)
                print(f"  student {shape:5s} seed {seed}: built in {st['built_in']}, home in {st['home_in']}", flush=True)
        p, opt = train(p, opt, builds, max(3, a.epochs // 4), memory, log, stage=f"dagger {r + 1}")
        save(p, log)
    print("saved", out + ".pkl")

    seeds = range(100, 101) if a.quick else range(100, 100 + a.eval_seeds)
    print("evaluating on unseen seeds (the tower was never trained on)...")
    rows = evaluate(p, memory, a.mult, seeds=seeds)
    rows += evaluate(p, memory, a.mult, seeds=seeds, who="teacher")
    json.dump(rows, open(out + "_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
