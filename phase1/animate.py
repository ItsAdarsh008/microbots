"""Animate the Phase 1 flock, live in a window or saved to a file.

Each dot is coloured by the direction it's heading, so bots that have lined
up with their neighbours share a colour. The beacon switches on halfway
through, and its target zone appears as a faint circle.

Examples (run from the repo root, with .venv active):
    python phase1/animate.py                                   # live window
    python phase1/animate.py --save media/flock.gif --size 4 --stride 3 --fps 20
    python phase1/animate.py --save media/flock.mp4 --size 10.8   # needs ffmpeg
"""

import argparse

import jax
import numpy as np

import flock


def main():
    parser = argparse.ArgumentParser(description="Animate the Phase 1 flock.")
    parser.add_argument("--bots", type=int, default=1000, help="number of bots")
    parser.add_argument("--steps", type=int, default=600, help="number of frames")
    parser.add_argument("--beacon-from", type=int, default=None,
                        help="frame the beacon switches on (default: halfway)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--stride", type=int, default=1,
                        help="show every Nth step (smaller files for GIFs)")
    parser.add_argument("--size", type=float, default=6.0, help="figure size in inches")
    parser.add_argument("--save", help="save to .gif or .mp4 instead of opening a window")
    args = parser.parse_args()
    beacon_from = args.steps // 2 if args.beacon_from is None else args.beacon_from

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    # 1. Simulate every frame first, then convert to NumPy for Matplotlib.
    traj_pos, traj_vel = flock.simulate(jax.random.PRNGKey(args.seed), args.bots,
                                        args.steps, beacon_from)
    traj_pos, traj_vel = np.asarray(traj_pos), np.asarray(traj_vel)
    # Heading angle of every bot in every frame, from -pi to pi.
    heading = np.arctan2(traj_vel[..., 1], traj_vel[..., 0])

    # 2. A square, black, axis-free picture.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, flock.BOX)
    ax.set_ylim(0, flock.BOX)
    ax.set_aspect("equal")
    ax.axis("off")

    zone = Circle(np.asarray(flock.TARGET), flock.TARGET_RADIUS, fill=False,
                  edgecolor="white", linewidth=1, linestyle="--", alpha=0.0)
    ax.add_patch(zone)
    dot_size = 6 if args.bots <= 500 else 4
    dots = ax.scatter(traj_pos[0, :, 0], traj_pos[0, :, 1], s=dot_size,
                      c=heading[0], cmap="hsv", vmin=-np.pi, vmax=np.pi)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white",
                    fontsize=10, va="top", alpha=0.8)

    # 3. Each frame: move the dots, recolour them, update the label and zone.
    def update(frame):
        dots.set_offsets(traj_pos[frame])
        dots.set_array(heading[frame])
        beacon_on = frame >= beacon_from
        zone.set_alpha(0.35 if beacon_on else 0.0)
        label.set_text("beacon on" if beacon_on else "free flocking")
        return dots, zone, label

    anim = FuncAnimation(fig, update, frames=range(0, args.steps, args.stride),
                         interval=1000 / args.fps, blit=True)

    if args.save:
        writer = "pillow" if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
