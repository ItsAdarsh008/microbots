"""Let the trained network drive the swarm and compare it with the rules.

    python linking/learn/evaluate.py                 # every shape, 5 seeds each
    python linking/learn/evaluate.py --seeds 3

Each free bot sees only its 5x5x5 block, the shape map for that block, and
the bot-to-bot work signal. The network scores its legal moves; the bot picks
one by chance in proportion to those scores (so it doesn't freeze in ties),
then falls back to the next best if that move would leave something
unattached. "Stay" is one of the choices.

Seeds 100+ are used here, so none of these builds were in the training data.
The tower was never trained on at all.
"""

import argparse
import os

import numpy as np

import env as E

HERE = os.path.dirname(__file__)


def load(path):
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("w")])
    return [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]


def logits(params, x):
    for w, b in params[:-1]:
        x = np.maximum(x @ w + b, 0.0)
    w, b = params[-1]
    return x @ w + b


def model_chooser(params, rng):
    def choose(sw, p, slots, ctx):
        obs = ctx["obs"]
        score = logits(params, obs)
        options = [0] + list(slots)
        s = score[options]
        prob = np.exp(s - s.max()); prob /= prob.sum()
        first = options[rng.choice(len(options), p=prob)]
        rest = [o for o in sorted(options, key=lambda o: -score[o]) if o != first]
        return [first] + rest
    return choose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(HERE, "policy.npz"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shapes", default="sheet,arch,tower")
    a = ap.parse_args()
    params = load(a.policy)
    print(f"{'shape':6s} {'who':8s} finished   steps (each run)")
    for shape in a.shapes.split(","):
        for who in ("rules", "network"):
            done, steps = 0, []
            for seed in range(100, 100 + a.seeds):
                if who == "rules":
                    chooser = E.rule_chooser
                else:
                    chooser = model_chooser(params, np.random.default_rng(seed))
                ok, n, sw = E.build(shape, chooser, seed=seed, max_steps=300)
                assert E.L.is_connected(sw.occ)
                done += ok; steps.append(n if ok else "-")
            print(f"{shape:6s} {who:8s} {done}/{a.seeds}        {steps}")


if __name__ == "__main__":
    main()
