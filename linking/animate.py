"""Render the linked-lattice swarm in 3D: cubes, their links, and every move.

Each bot is drawn as a cube. Teal bars between cubes are links (shared
faces). A bot that slides glides one cell; a bot that pivots rolls half a
turn over the edge of the bot underneath it, from that bot's top face down
onto its side face, the way a cube tumbles over a corner.

Colours: purple = locked into the shape, grey = free (crawling or parked),
orange = pivoting over an edge right now, red = removed by damage. Faint dots
show target cells that are still empty.

Examples (run from the repo root, with .venv active):
    python linking/animate.py --shape sheet                     # live window
    python linking/animate.py --shape arch --save media/captures/linking_arch.mp4
    python linking/animate.py --shape sheet --repair --save media/captures/linking_sheet_repair.mp4
    python linking/animate.py --shape arch --save media/linking_arch.gif --sub 2 --size 4.5
"""

import argparse

import numpy as np

import lattice as L

CUBE = 0.72                    # drawn cube size (1.0 would leave no gap)
COLOURS = {
    "locked": "#7b5cd6",
    "free": "#a7adb5",
    "pivot": "#f39c34",
    "lost": "#e04848",
    "link": "#1fb5a8",
}

# The repair demo for each shape: which cells get cut, and how many spares.
REPAIRS = {
    "sheet": (lambda: L.cut_top("sheet", 5), 40),
    "arch": (lambda: L.cut_box(14, 18, 0, 16, 0, 14), 30),
    "tower": (lambda: L.cut_box(13, 19, 0, 5, 3, 14), 40),
}

# The 8 corners and 6 faces of a unit cube centred on the origin.
_CORNERS = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                     for z in (-.5, .5)])
_FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
          [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def rotation(axis, angle):
    """Rotation matrix for turning `angle` radians around unit vector `axis`."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def ease(s):
    """Smooth start and stop for each move."""
    return s * s * (3 - 2 * s)


def bot_pose(start, end, via, kind, s):
    """Centre and rotation of one bot, a fraction s through its move."""
    start, end, via = (np.asarray(v, dtype=float) for v in (start, end, via))
    if kind == 1:                                   # slide: straight line
        return start + (end - start) * s, np.eye(3)
    if kind == 2:                                   # pivot: half-turn over an edge
        hinge = (start + end) / 2                   # the edge it rolls over
        v0 = start - hinge
        v_half = via - hinge                        # where it is halfway round
        axis = np.cross(v0, v_half)
        axis /= np.linalg.norm(axis)
        turn = rotation(axis, s * np.pi)
        return hinge + turn @ v0, turn
    return end, np.eye(3)


def cube_faces(centre, turn):
    corners = centre + (_CORNERS * CUBE) @ turn.T
    return [corners[f] for f in _FACES]


def link_segments(centres):
    """Short bars across the gap between every pair of face-touching bots."""
    cells = {tuple(np.round(c).astype(int)): c for c in centres}
    segs = []
    for cell, c in cells.items():
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            other = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
            if other in cells:
                d = np.array(d)
                segs.append([c + d * CUBE / 2 - d * 0.02, c + d * (1 - CUBE / 2) + d * 0.02])
    return segs


def main():
    parser = argparse.ArgumentParser(description="Render the linked lattice swarm.")
    parser.add_argument("--shape", default="sheet", choices=list(L.SHAPES))
    parser.add_argument("--repair", action="store_true",
                        help="after building, cut part of the shape and rebuild it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sub", type=int, default=5,
                        help="animation frames per simulation step (smoothness)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=float, default=7.2, help="figure size in inches")
    parser.add_argument("--hold", type=int, default=45, help="frames to hold at the end")
    parser.add_argument("--save", help="save to .gif or .mp4 instead of opening a window")
    args = parser.parse_args()

    if args.repair:
        make_region, extra = REPAIRS[args.shape]
        target, frames, swarm = L.run(args.shape, seed=args.seed,
                                      damage_region=make_region(), extra=extra)
    else:
        target, frames, swarm = L.run(args.shape, seed=args.seed)
    print(f"{len(frames)} steps, {swarm.moves_made} moves, "
          f"always connected: {all(f['connected'] for f in frames)}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    fig = plt.figure(figsize=(args.size, args.size * 0.75), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    nx, ny, nz = L.GRID
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(-1, nz - 3)
    ax.set_box_aspect((nx, ny, nz - 2), zoom=1.45)
    ax.computed_zorder = False      # draw links on top so they stay visible

    # Floor grid, drawn once.
    floor = []
    for x in range(nx + 1):
        floor.append([(x - .5, -.5, -.5), (x - .5, ny - .5, -.5)])
    for y in range(ny + 1):
        floor.append([(-.5, y - .5, -.5), (nx - .5, y - .5, -.5)])
    ax.add_collection3d(Line3DCollection(floor, colors="#e3e6ea", linewidths=0.5))

    cubes = Poly3DCollection([], edgecolors="#3a3f47", linewidths=0.3, zorder=1)
    ax.add_collection3d(cubes, autolim=False)
    links = Line3DCollection([], colors=COLOURS["link"], linewidths=1.6, alpha=0.9,
                             zorder=2)
    ax.add_collection3d(links, autolim=False)
    ghost = ax.scatter([], [], [], s=6, c="#7b5cd6", alpha=0.25, depthshade=False)
    title = fig.text(0.03, 0.95, "", fontsize=12, weight="bold", color="#222")
    status = fig.text(0.03, 0.91, "", fontsize=9.5, color="#555")

    damaged_tick = next((i for i, f in enumerate(frames) if f["damaged"]), None)
    lost_ids = None
    if damaged_tick is not None:
        lost_ids = frames[damaged_tick - 1]["alive"] & ~frames[damaged_tick]["alive"]

    # Each simulation step gets `sub` animation frames, except while spares
    # are walking back to the parking area, which plays faster.
    finished = next((i for i, f in enumerate(frames) if f["filled"] >= 1.0), len(frames))
    schedule = []
    for t in range(len(frames)):
        parking = finished < t and (damaged_tick is None or t < damaged_tick)
        n = max(1, args.sub // 3) if parking else args.sub
        schedule += [(t, (k + 1) / n) for k in range(n)]
    schedule += [(len(frames) - 1, 1.0)] * args.hold

    def stage(t):
        f = frames[t]
        if damaged_tick is not None and t >= damaged_tick:
            if f["filled"] >= 1.0:
                return "Repaired"
            if t < damaged_tick + 2:
                return f"Damage: {f['lost']} bots removed"
            return "Repairing: spares crawl back up"
        if f["filled"] >= 1.0:
            return "Built" if damaged_tick is None else "Built, spares parking"
        return "Building: crawling over each other, never letting go"

    def update(i):
        t, s = schedule[i]
        f = frames[t]
        s = ease(s)
        faces, colours, still = [], [], []
        for b in range(len(f["pos"])):
            if not f["alive"][b]:
                if lost_ids is not None and lost_ids[b] and t == damaged_tick:
                    # Show removed bots dropping away during the damage step.
                    centre = f["pos"][b] + np.array([0, 0, -3 * s])
                    faces += cube_faces(centre, np.eye(3))
                    colours += [COLOURS["lost"]] * 6
                continue
            kind = f["kind"][b]
            centre, turn = bot_pose(f["from"][b], f["pos"][b], f["via"][b], kind, s)
            faces += cube_faces(centre, turn)
            if kind == 2:
                colour = COLOURS["pivot"]
            elif f["bot_locked"][b]:
                colour = COLOURS["locked"]
            else:
                colour = COLOURS["free"]
            if not kind:
                still.append(centre)          # links are drawn between still bots
            colours += [colour] * 6
        cubes.set_verts(faces)
        cubes.set_facecolor(colours)
        links.set_segments(link_segments(still))
        empty = np.argwhere(target & ~f["occ"])
        ghost._offsets3d = (empty[:, 0], empty[:, 1], empty[:, 2]) if len(empty) else ([], [], [])
        ax.view_init(elev=22, azim=-60 + 28 * np.sin(i / 160))   # slow sway
        title.set_text(f"{args.shape.title()} · {stage(t)}")
        n_alive = int(f["alive"].sum())
        status.set_text(f"step {t}   bots {n_alive}   links {L.count_links(f['occ'])}   "
                        f"shape filled {f['filled']:.0%}   one connected piece: "
                        f"{'yes' if f['connected'] else 'NO'}")
        return cubes, links, ghost, title, status

    anim = FuncAnimation(fig, update, frames=len(schedule),
                         interval=1000 / args.fps, blit=False)
    if args.save:
        writer = "pillow" if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps, dpi=150 if writer == "ffmpeg" else 80)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
