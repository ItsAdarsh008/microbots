"""Animate Phase 2 self-assembly, live in a window or saved to a file.

The target shape is drawn faintly. Robots are coloured by state:
grey = waiting, orange = moving outside the shape, yellow = moving inside it,
purple = joined, green = seed.

Examples (run from the repo root, with .venv active):
    python phase2/animate.py --shape star
    python phase2/animate.py --shape wall --save media/assemble_wall.gif --size 4
    python phase2/animate.py --shape k --save media/assemble_k.mp4 --size 10.8   # needs ffmpeg
"""

import argparse

import numpy as np

import assemble

COLOURS = {assemble.WAIT: "#6b6b6b", assemble.MOVE_OUT: "#ff9a3c",
           assemble.MOVE_IN: "#ffe14d", assemble.JOINED: "#b98cff"}
SEED_COLOUR = "#5ee38c"


def main():
    parser = argparse.ArgumentParser(description="Animate Phase 2 self-assembly.")
    parser.add_argument("--shape", default="star", choices=list(assemble.SHAPES))
    parser.add_argument("--movers", type=int, default=150, help="robots that move (plus 4 seeds)")
    parser.add_argument("--steps", type=int, default=7000, help="simulation steps")
    parser.add_argument("--every", type=int, default=20, help="steps per animation frame")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--size", type=float, default=6.0, help="figure size in inches")
    parser.add_argument("--save", help="save to .gif or .mp4 instead of opening a window")
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    print("simulating...")
    mask, traj_pos, traj_state, _, is_seed = assemble.run(
        args.shape, args.movers, args.steps, args.every)
    n_frames = traj_pos.shape[0]

    # Colour every robot in every frame from its state.
    palette = np.array([COLOURS[s] for s in range(4)])
    colours = palette[traj_state]
    colours[:, is_seed] = SEED_COLOUR

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect("equal")
    ax.axis("off")

    # The target shape, drawn faintly behind the robots.
    extent = [assemble.ORIGIN[0], assemble.ORIGIN[0] + assemble.GRID_SIZE * assemble.RES,
              assemble.ORIGIN[1], assemble.ORIGIN[1] + assemble.GRID_SIZE * assemble.RES]
    ax.imshow(mask.T, origin="lower", extent=extent, cmap="Greys_r", alpha=0.18,
              interpolation="nearest")

    # Frame the whole scene: the starting cluster and the shape.
    xs, ys = np.where(mask)
    all_x = np.concatenate([traj_pos[..., 0].ravel(), extent[0] + (xs + 0.5) * assemble.RES])
    all_y = np.concatenate([traj_pos[..., 1].ravel(), extent[2] + (ys + 0.5) * assemble.RES])
    pad, span = 1.5, max(np.ptp(all_x), np.ptp(all_y)) / 2 + 1.5
    cx, cy = (all_x.min() + all_x.max()) / 2, (all_y.min() + all_y.max()) / 2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)

    # Dot size so each robot is about one robot-diameter wide on screen.
    pts_per_unit = args.size * 72 / (2 * span)
    dots = ax.scatter(traj_pos[0, :, 0], traj_pos[0, :, 1], s=(0.85 * pts_per_unit) ** 2,
                      c=colours[0], linewidths=0)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white",
                    fontsize=10, va="top", alpha=0.8)

    def update(frame):
        dots.set_offsets(traj_pos[frame])
        dots.set_color(colours[frame])
        joined = int((traj_state[frame] == assemble.JOINED).sum() - is_seed.sum())
        label.set_text(f"{args.shape} · {joined}/{args.movers} joined")
        return dots, label

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=True)

    if args.save:
        writer = "pillow" if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
