"""Contact sheet of the held-out shapes with their sampled particles, labelled with the genus.

    python lab/preview_shapes.py [out.png] [--n 1024] [--train]

--train draws the training shapes instead (the 8 train emoji and 17 procedural seeds from 1000).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agnostic as ag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="/tmp/agnostic_shapes.png")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    if args.train:
        suite = []
        for name in ag.TRAIN_EMOJI:
            m, r = ag.emoji_shape(name)
            suite.append({"name": name, "genus": ag.genus(m), "mask": m, "rgb": r})
        for sd in range(1000, 1017):
            m, r = ag.proc_shape(sd)
            suite.append({"name": f"proc:{sd}", "genus": ag.genus(m), "mask": m, "rgb": r})
    else:
        suite = ag.heldout_suite()
    cols = 5
    rows = -(-len(suite) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sh in zip(axes.ravel(), suite):
        x, s = ag.sample_swarm(sh["mask"], sh["rgb"], args.n, 7)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec="0.8", lw=0.8))
        ax.scatter(x[:, 0], x[:, 1], c=s[:, -3:] + 0.5, s=6, linewidths=0)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{sh['name']}   genus {sh['genus']}   n={len(x)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
