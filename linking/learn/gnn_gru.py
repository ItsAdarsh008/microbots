"""A graph neural network with GRU memory, as the bots' shared policy.

The first trained policy (train_bc.py, dagger.py) is an MLP, and it was handed
two inputs written by hand: a "how far to the work" signal relayed bot to bot,
and a compass pointing at the nearest open cell. Without them it can't find
the work. This file asks: can the bots learn to do that relaying themselves?

    * GNN: every step, each bot sends a message to the bots around it (the 26
      cells touching it, faces, edges and corners) and combines what it hears,
      a few rounds per step. That's how information moves through the swarm.
    * GRU: each bot keeps a hidden state between steps and updates it with a
      GRU cell. That's memory, so information can keep travelling across many
      steps, the way the Kilobots' hop-count gradient settled over time.

Each bot sees only its own 5x5x5 block (which cells hold a bot, which belong
to the shape) and whether it's locked: no relayed signal, no compass. Every
bot runs the same network.

Training is imitation, as before: record the hand-written rules (the
teacher), train the network to predict the teacher's move, then DAgger
(the network drives, the teacher labels). The teacher uses a map of the whole
grid that the bots can't see, so to copy it well the network has to work out
"which way is the work" from what its neighbours tell it.

    python linking/learn/gnn_gru.py                 # record, train, DAgger, evaluate
    python linking/learn/gnn_gru.py --quick         # small smoke test
    python linking/learn/gnn_gru.py --no-memory     # same, but the GRU state is wiped every step

Runs on a CPU; a GPU (Colab) makes the training part much faster. The data
recording and the evaluation run the Python simulation, which is CPU only.
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

N_FEAT = 2 * E.SIDE ** 3 + 1          # 125 occupied + 125 shape + 1 locked = 251
NEIGHBOURS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                       if (a, b, c) != (0, 0, 0)])                # 26 cells touching a bot
N_PAD = 256                            # bots per build, padded (the biggest training build has ~180)
E_PAD = N_PAD * len(NEIGHBOURS)        # message slots per step
WINDOW = 32                            # steps per training window (truncated backprop through time)


# ============================================================== recording

def features(sw):
    """What each bot sees (251 numbers each), and its own cell, for every bot."""
    goal = sw.goal_cells()
    frame = E.frame_state(sw, goal)
    n = len(sw.pos)
    x = np.zeros((N_PAD, N_FEAT), dtype=np.uint8)          # all 0/1, stored small
    for b in range(n):
        if not sw.alive[b]:
            continue
        obs = E.observe(frame, tuple(int(v) for v in sw.pos[b]))
        x[b, :250] = obs[:250]
        x[b, 250] = obs[257]                       # locked
    return x


def edges(sw):
    """Who can hear whom this step: every pair of bots in touching cells.

    Returned as (senders, receivers, offsets) padded to E_PAD. Padding points at
    the spare node N_PAD - 1, which never holds a bot.
    """
    pos = np.asarray(sw.pos)
    n = len(pos)
    snd, rcv, off = [], [], []
    for d in NEIGHBOURS:
        q = pos + d
        ok = np.all((q >= 0) & (q < np.array(L.GRID)), axis=1) & sw.alive[:n]
        idx = np.full(n, -1)
        idx[ok] = sw.id_at[tuple(q[ok].T)]
        has = idx >= 0
        has &= np.where(has, sw.alive[np.maximum(idx, 0)], False)
        snd.append(idx[has])
        rcv.append(np.nonzero(has)[0])
        off.append(np.repeat(d[None, :], has.sum(), axis=0))
    snd, rcv, off = np.concatenate(snd), np.concatenate(rcv), np.concatenate(off)
    m = len(snd)
    S = np.full(E_PAD, N_PAD - 1, dtype=np.int16)
    R = np.full(E_PAD, N_PAD - 1, dtype=np.int16)
    O = np.zeros((E_PAD, 3), dtype=np.int8)
    S[:m], R[:m], O[:m] = snd, rcv, off
    return S, R, O


def teacher_labels(sw):
    """At the start of a step: for every free bot, its legal moves and the teacher's pick."""
    goal = sw.goal_cells()
    dist = L.distance_field(sw.occ, sw.locked, goal)
    busy = sw.occ.copy()
    mask = np.zeros((N_PAD, E.N_ACTIONS), dtype=bool)
    mask[:, 0] = True
    label = np.zeros(N_PAD, dtype=np.int32)
    train = np.zeros(N_PAD, dtype=bool)
    for b in range(len(sw.pos)):
        p = tuple(int(v) for v in sw.pos[b])
        if not sw.alive[b] or sw.locked[p]:
            continue
        slots = E.legal_slots(sw, p, busy)
        for s in slots:
            mask[b, s] = True
        order = E.rule_order(sw, p, slots, dist)
        label[b] = order[0] if order else 0
        train[b] = True
    return mask, label, train


def rollout(shape, seed, driver=None, max_steps=200, extra=12):
    """One build, recording every step. driver=None: the teacher drives; else a Driver."""
    sw = L.Swarm(L.SHAPES[shape](), seed=seed, extra=extra)
    assert len(sw.pos) < N_PAD
    rec = {k: [] for k in ("x", "snd", "rcv", "off", "mask", "label", "train")}
    if driver is not None:
        driver.reset()
    t = 0
    for t in range(max_steps):
        if sw.done():
            break
        x = features(sw)
        S, R, O = edges(sw)
        mask, label, train = teacher_labels(sw)
        for k, v in zip(rec, (x, S, R, O, mask, label, train)):
            rec[k].append(v)
        if driver is None:
            E.step(sw, E.rule_chooser)
        else:
            driver.step(sw, x, S, R, O, mask)
    return {k: np.stack(v) for k, v in rec.items()}, sw.done(), t


# ============================================================== the network

H = 64          # hidden size
ROUNDS = 3      # message-passing rounds per step


def dense(key, n_in, n_out):
    return {"w": jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in), "b": jnp.zeros(n_out)}


def init(key):
    ks = iter(jax.random.split(key, 32))
    p = {
        "enc1": dense(next(ks), N_FEAT, H), "enc2": dense(next(ks), H, H),
        "from_mem": dense(next(ks), H, H),
        "msg": [dense(next(ks), 2 * H + 3, H) for _ in range(ROUNDS)],
        "upd": [dense(next(ks), 2 * H, H) for _ in range(ROUNDS)],
        "head1": dense(next(ks), 2 * H, H), "head2": dense(next(ks), H, E.N_ACTIONS),
        # GRU cell: update gate z, reset gate r, candidate n
        "gru_x": dense(next(ks), H, 3 * H), "gru_h": dense(next(ks), H, 3 * H),
    }
    return p


def lin(p, x):
    return x @ p["w"] + p["b"]


def gru(p, h, x):
    """One GRU step. z: how much to update; r: how much old memory to use for the candidate."""
    gx, gh = lin(p["gru_x"], x), lin(p["gru_h"], h)
    zx, rx, nx = jnp.split(gx, 3, axis=-1)
    zh, rh, nh = jnp.split(gh, 3, axis=-1)
    z = jax.nn.sigmoid(zx + zh)
    r = jax.nn.sigmoid(rx + rh)
    n = jnp.tanh(nx + r * nh)
    return (1 - z) * n + z * h


def step_fn(p, h, x, snd, rcv, off, memory=True):
    """One simulation step for every bot: listen to neighbours, update memory, score moves."""
    x, off = x.astype(jnp.float32), off.astype(jnp.float32)
    snd, rcv = snd.astype(jnp.int32), rcv.astype(jnp.int32)
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
    logits = lin(p["head2"], jax.nn.relu(lin(p["head1"], jnp.concatenate([u, h2], axis=-1))))
    return h2, logits


def count(p):
    return sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(p))


# ============================================================== training

def window_loss(p, h0, w, memory):
    """Cross-entropy over one window of steps for one build, carrying the GRU state through it."""
    def body(h, s):
        h, logits = step_fn(p, h, s["x"], s["snd"], s["rcv"], s["off"], memory)
        logits = jnp.where(s["mask"], logits, -1e9)
        logp = jax.nn.log_softmax(logits)
        nll = -jnp.take_along_axis(logp, s["label"][:, None], axis=1)[:, 0]
        hit = (jnp.argmax(logits, axis=1) == s["label"])
        return h, (jnp.sum(nll * s["train"]), jnp.sum(hit * s["train"]), jnp.sum(s["train"]))
    h, (nll, hit, n) = jax.lax.scan(body, h0, w)
    total = jnp.maximum(n.sum(), 1.0)
    return nll.sum() / total, (h, hit.sum() / total)


def batched_loss(p, h0, wb, memory):
    loss, (h, acc) = jax.vmap(window_loss, in_axes=(None, 0, 0, None))(p, h0, wb, memory)
    return loss.mean(), (h, acc.mean())


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": 0}


def make_train_step(memory):
    @jax.jit
    def train_step(p, opt, h0, wb, lr):
        (loss, (h, acc)), g = jax.value_and_grad(batched_loss, has_aux=True)(p, h0, wb, memory)
        g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), g)
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, opt["m"], g)
        v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, opt["v"], g)
        p = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, {"m": m, "v": v, "t": t}, jax.lax.stop_gradient(h), loss, acc
    return train_step


def pad_build(rec, T):
    """Pad one recorded build to T steps (extra steps train nothing)."""
    t = len(rec["x"])
    out = {}
    for k, v in rec.items():
        pad = [(0, T - t)] + [(0, 0)] * (v.ndim - 1)
        fill = N_PAD - 1 if k in ("snd", "rcv") else 0
        out[k] = np.pad(v, pad, constant_values=fill)
    out["mask"][t:, :, 0] = True
    return out


def stack(builds):
    T = WINDOW * int(np.ceil(max(len(b["x"]) for b in builds) / WINDOW))
    padded = [pad_build(b, T) for b in builds]
    return {k: np.stack([b[k] for b in padded]) for k in padded[0]}, T


def train(p, opt, builds, epochs, memory, log, batch=4, lr=1e-3, stage="imitation"):
    data, T = stack(builds)
    step = make_train_step(memory)
    rng = np.random.default_rng(len(log["loss"]))
    nb = len(builds)
    for ep in range(epochs):
        order = rng.permutation(nb)
        ls, accs = [], []
        for i in range(0, nb - batch + 1, batch):
            idx = order[i:i + batch]
            h = jnp.zeros((batch, N_PAD, H))
            for w0 in range(0, T, WINDOW):
                wb = {k: jnp.asarray(v[idx, w0:w0 + WINDOW]) for k, v in data.items()}
                if not np.any(np.asarray(wb["train"])):
                    break
                p, opt, h, loss, acc = step(p, opt, h, wb, lr)
                ls.append(float(loss)); accs.append(float(acc))
                log["loss"].append(round(float(loss), 4))
        log["epoch_end"].append(len(log["loss"])); log["agree"].append(round(float(np.mean(accs)), 4))
        log["stage"].append(stage)
        print(f"  {stage:9s} epoch {len(log['agree']):3d}  loss {np.mean(ls):.3f}  "
              f"picks the teacher's move {np.mean(accs):.1%}", flush=True)
    return p, opt


# ============================================================== driving the swarm

class Driver:
    """The trained network driving every free bot, with a GRU state per bot."""

    def __init__(self, p, memory=True, seed=0):
        self.p, self.memory = p, memory
        self.rng = np.random.default_rng(seed)
        self.fwd = jax.jit(lambda p, h, x, s, r, o: step_fn(p, h, x, s, r, o, memory))

    def reset(self):
        self.h = jnp.zeros((N_PAD, H))

    def step(self, sw, x, S, R, O, mask):
        self.h, logits = self.fwd(self.p, self.h, jnp.asarray(x), jnp.asarray(S), jnp.asarray(R), jnp.asarray(O))
        logits = np.asarray(logits)

        def chooser(sw_, pcell, slots, ctx):
            b = sw_.id_at[pcell]
            options = [0] + list(slots)
            s = logits[b, options]
            prob = np.exp(s - s.max()); prob /= prob.sum()
            first = options[self.rng.choice(len(options), p=prob)]
            rest = [o for o in sorted(options, key=lambda o: -logits[b, o]) if o != first]
            return [first] + rest
        E.step(sw, chooser)


def evaluate(p, memory, shapes=("sheet", "arch", "tower"), seeds=range(100, 105), max_steps=300):
    out = {}
    for shape in shapes:
        res = []
        for seed in seeds:
            sw = L.Swarm(L.SHAPES[shape](), seed=seed, extra=12)
            d = Driver(p, memory, seed)
            d.reset()
            t = 0
            for t in range(max_steps):
                if sw.done():
                    break
                x = features(sw)
                S, R, O = edges(sw)
                d.step(sw, x, S, R, O, None)
            assert L.is_connected(sw.occ)
            res.append(t if sw.done() else None)
        out[shape] = res
        done = [r for r in res if r is not None]
        print(f"  {shape:6s} {len(done)}/{len(res)} built   steps: {res}", flush=True)
    return out


# ============================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--no-memory", action="store_true", help="wipe the GRU state every step (ablation)")
    ap.add_argument("--seeds", type=int, default=12, help="teacher builds per training shape")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dagger", type=int, default=2, help="DAgger rounds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    memory = not a.no_memory
    if a.quick:
        a.seeds, a.epochs, a.dagger = 2, 2, 0
    name = "gnn_gru" if memory else "gnn_nomem"
    out = a.out or os.path.join(HERE, name)
    shapes = ["sheet", "arch"]
    print("devices:", jax.devices())

    t0 = time.time()
    print("recording the teacher...")
    builds = []
    for shape in shapes:
        for seed in range(a.seeds):
            rec, ok, steps = rollout(shape, seed)
            builds.append(rec)
            print(f"  teacher {shape:5s} seed {seed:2d}: {steps} steps, finished={ok}", flush=True)
    print(f"recorded {len(builds)} builds in {time.time() - t0:.0f}s")

    p = init(jax.random.PRNGKey(0))
    opt = adam_init(p)
    print(f"network: {count(p):,} weights, {ROUNDS} message rounds per step, GRU memory {'on' if memory else 'off'}")
    log = {"loss": [], "epoch_end": [], "agree": [], "stage": []}
    t0 = time.time()
    p, opt = train(p, opt, builds, a.epochs, memory, log)
    print(f"imitation done in {time.time() - t0:.0f}s")

    for r in range(a.dagger):
        print(f"DAgger round {r + 1}: the network drives, the teacher labels")
        new = []
        for shape in shapes:
            for seed in range(1000 + 10 * r, 1000 + 10 * r + a.seeds // 2):
                rec, ok, steps = rollout(shape, seed, Driver(p, memory, seed), max_steps=150)
                new.append(rec)
                print(f"  student {shape:5s} seed {seed}: {steps} steps, finished={ok}", flush=True)
        builds += new
        p, opt = train(p, opt, builds, max(4, a.epochs // 4), memory, log, stage=f"dagger {r + 1}")

    with open(out + ".pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    json.dump(log, open(out + "_log.json", "w"))
    print("saved", out + ".pkl")

    print("evaluating on unseen seeds (the tower was never trained on)...")
    res = evaluate(p, memory, seeds=range(100, 102) if a.quick else range(100, 105))
    json.dump(res, open(out + "_eval.json", "w"))


if __name__ == "__main__":
    main()
