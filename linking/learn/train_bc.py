"""Stage 1 training: imitation (behaviour cloning).

1. Run the hand-written rules (the teacher) on many builds and record, for
   every free bot at every step, what it saw (262 numbers), which moves were
   legal, and which move it ended up making.
2. Train one small shared network to predict that move from what the bot saw.
   Illegal moves are masked out, so it only ever chooses among legal ones.

    python linking/learn/train_bc.py                # record data, train, save policy.npz
    python linking/learn/train_bc.py --seeds 4      # quicker

The network: 262 -> 128 -> 128 -> 19, ReLU, about 52,600 weights.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import env as E

HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------- data

def record(shapes, seeds):
    obs, masks, labels = [], [], []
    for shape in shapes:
        for seed in seeds:
            rec = []
            ok, steps, _ = E.build(shape, E.rule_chooser, seed=seed, record=rec)
            for o, m, a in rec:
                obs.append(o); masks.append(m); labels.append(a)
            print(f"  teacher {shape:5s} seed {seed:3d}: finished={ok} in {steps} steps, {len(rec)} decisions")
    return np.stack(obs), np.stack(masks), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------- network

def init(key, sizes=(E.OBS_SIZE, 128, 128, E.N_ACTIONS)):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append({"w": jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
                       "b": jnp.zeros(n_out)})
    return params


def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def masked_logits(params, x, mask):
    return jnp.where(mask, forward(params, x), -1e9)


def loss_fn(params, x, mask, y):
    logp = jax.nn.log_softmax(masked_logits(params, x, mask))
    return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=1))


# Adam, written out (no optimiser library needed)
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


@jax.jit
def train_step(params, opt, x, mask, y, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    loss, g = jax.value_and_grad(loss_fn)(params, x, mask, y)
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], g)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}, loss


@jax.jit
def accuracy(params, x, mask, y):
    return jnp.mean(jnp.argmax(masked_logits(params, x, mask), axis=1) == y)


def save(params, path):
    flat = {}
    for i, layer in enumerate(params):
        flat[f"w{i}"] = np.asarray(layer["w"]); flat[f"b{i}"] = np.asarray(layer["b"])
    np.savez(path, **flat)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shapes", default="sheet,arch")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "policy.npz"))
    a = ap.parse_args()

    shapes = a.train_shapes.split(",")
    t0 = time.time()
    print("recording the teacher...")
    x, m, y = record(shapes, range(a.seeds))
    print(f"{len(y)} decisions in {time.time() - t0:.0f}s; {np.mean(y > 0):.0%} are moves, the rest are 'stay'")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = len(y) // 10
    val, tr = order[:n_val], order[n_val:]
    params = init(jax.random.PRNGKey(0))
    opt = adam_init(params)
    batch = 512
    for epoch in range(a.epochs):
        rng.shuffle(tr)
        losses = []
        for i in range(0, len(tr) - batch + 1, batch):
            idx = tr[i:i + batch]
            params, opt, loss = train_step(params, opt, x[idx], m[idx], y[idx])
            losses.append(float(loss))
        acc = float(accuracy(params, x[val], m[val], y[val]))
        print(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.3f}  agrees with the teacher on {acc:.1%} of held-out decisions")
    save(params, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
