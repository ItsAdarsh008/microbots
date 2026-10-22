"""Animate the JAX random walk, live in a window or saved to a file.

Examples (run from the repo root, with .venv active):
    python phase0/animate.py                              # live window, 100 bots
    python phase0/animate.py --bots 1000                  # live window, 1,000 bots
    python phase0/animate.py --bots 1000 --save media/random_walk.gif
    python phase0/animate.py --bots 1000 --save media/random_walk.mp4   # needs ffmpeg
"""

import argparse

import jax
import numpy as np

from random_walk_jax import BOX, simulate

DOT_COLOR = "#b98cff"  # soft purple on black


def main():
    parser = argparse.ArgumentParser(description="Animate the Phase 0 random walk.")
    parser.add_argument("--bots", type=int, default=100, help="number of bots")
    parser.add_argument("--steps", type=int, default=300, help="number of frames")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--size", type=float, default=6.0, help="figure size in inches")
    parser.add_argument("--save", help="save to .gif or .mp4 instead of opening a window")
    args = parser.parse_args()

    # When saving, draw off-screen. This must happen before pyplot is imported.
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # 1. Simulate every frame first (JAX), then convert to NumPy for Matplotlib.
    traj = np.asarray(simulate(jax.random.PRNGKey(args.seed), args.bots, args.steps))

    # 2. Set up one square, black, axis-free picture.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, BOX)
    ax.set_ylim(0, BOX)
    ax.set_aspect("equal")
    ax.axis("off")
    dot_size = 6 if args.bots <= 300 else 3
    dots = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=dot_size, c=DOT_COLOR)

    # 3. For each frame, move every dot to that frame's positions.
    def update(frame):
        dots.set_offsets(traj[frame])
        return (dots,)

    anim = FuncAnimation(fig, update, frames=args.steps,
                         interval=1000 / args.fps, blit=True)

    if args.save:
        writer = "pillow" if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
