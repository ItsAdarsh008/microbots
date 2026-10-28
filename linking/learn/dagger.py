"""Stage 1b: DAgger ("dataset aggregation").

Pure copying has a weakness: once the student makes a mistake it ends up in
situations the teacher never showed it, and errors pile up. DAgger fixes this:

1. Let the trained network drive the swarm.
2. At every decision, also ask the teacher what IT would have done there.
3. Add those (what the bot saw, teacher's move) pairs to the training data.
4. Retrain. Repeat.

    python linking/learn/dagger.py --rounds 2
"""

import argparse
import os

import jax
import numpy as np

import env as E
import evaluate as V
import train_bc as T

HERE = os.path.dirname(__file__)


def teacher_label(sw, p, slots, ctx):
    order = E.rule_order(sw, p, slots, ctx["dist"])
    return order[0] if order else 0


def collect(params, shapes, seeds):
    """Student drives; the teacher labels every decision."""
    obs, masks, labels = [], [], []
    for shape in shapes:
        for seed in seeds:
            student = V.model_chooser(params, np.random.default_rng(seed))

            def chooser(sw, p, slots, ctx):
                m = np.zeros(E.N_ACTIONS, dtype=bool); m[0] = True
                for s in slots: m[s] = True
                obs.append(ctx["obs"]); masks.append(m); labels.append(teacher_label(sw, p, slots, ctx))
                return student(sw, p, slots, ctx)

            ok, n, _ = E.build(shape, chooser, seed=seed, max_steps=150, record=[])
    return np.stack(obs), np.stack(masks), np.array(labels, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--train-shapes", default="sheet,arch")
    ap.add_argument("--out", default=os.path.join(HERE, "policy.npz"))
    a = ap.parse_args()
    shapes = a.train_shapes.split(",")
    print("recording the teacher...")
    x, m, y = T.record(shapes, range(a.seeds))
    params = T.init(jax.random.PRNGKey(0)); opt = T.adam_init(params)
    rng = np.random.default_rng(0)

    def fit(params, opt, x, m, y, epochs):
        for _ in range(epochs):
            idx_all = rng.permutation(len(y))
            for i in range(0, len(y) - 511, 512):
                idx = idx_all[i:i + 512]
                params, opt, _ = T.train_step(params, opt, x[idx], m[idx], y[idx])
        return params, opt

    params, opt = fit(params, opt, x, m, y, 10)
    for r in range(a.rounds):
        np_params = [(np.asarray(l["w"]), np.asarray(l["b"])) for l in params]
        xs, ms, ys = collect(np_params, shapes, range(1000 + 10 * r, 1000 + 10 * r + a.seeds))
        x, m, y = np.concatenate([x, xs]), np.concatenate([m, ms]), np.concatenate([y, ys])
        params, opt = fit(params, opt, x, m, y, 6)
        print(f"round {r + 1}: +{len(ys)} decisions where the student drove and the teacher labelled ({len(y)} total)")
    T.save(params, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
