"""Draw the move rules as a side-view diagram: media/linking_moves.png

    python linking/diagram.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

PURPLE, GREY, ORANGE, RED, TEAL = "#7b5cd6", "#a7adb5", "#f39c34", "#e04848", "#1fb5a8"


def square(ax, x, z, colour, dashed=False, alpha=1.0):
    ax.add_patch(Rectangle((x + .06, z + .06), .88, .88, facecolor="none" if dashed else colour,
                           edgecolor=colour if dashed else "#3a3f47",
                           linestyle="--" if dashed else "-", lw=1.4, alpha=alpha))


def floor(ax, x0, x1):
    ax.plot([x0, x1], [0, 0], color="#888", lw=2)
    for x in np.arange(x0, x1, .3):
        ax.plot([x, x - .2], [0, -.2], color="#bbb", lw=.8)


def arrow(ax, a, b, rad=0.0, colour=ORANGE):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=14, lw=2, color=colour))


def panel(ax, title, note):
    ax.set_title(title, fontsize=11, weight="bold", loc="left")
    ax.text(0, -.75, note, fontsize=8.5, color="#444", va="top", wrap=True)
    ax.set_xlim(-.2, 4.2)
    ax.set_ylim(-1.6, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))

# 1. Slide
ax = axes[0]
floor(ax, -.2, 4.2)
for x in range(4):
    square(ax, x, 0, PURPLE)
square(ax, 1, 1, ORANGE)
square(ax, 2, 1, ORANGE, dashed=True)
arrow(ax, (1.5, 1.5), (2.5, 1.5))
panel(ax, "Slide", "Glide one cell along a surface.\nNeeds something solid under\nboth the start and end cell.")

# 2. Pivot
ax = axes[1]
floor(ax, -.2, 4.2)
square(ax, 1, 0, PURPLE)
square(ax, 1, 1, ORANGE)
square(ax, 2, 1, ORANGE, dashed=True, alpha=.45)
square(ax, 2, 0, ORANGE, dashed=True)
ax.plot(2, 1, "o", color="#222", ms=6)
ax.text(2.08, .78, "hinge", fontsize=8)
arrow(ax, (1.5, 1.6), (2.6, .45), rad=-0.75)
panel(ax, "Pivot", "Roll half a turn over the edge of\nthe bot underneath (the hinge),\nswinging through the corner cell.")

# 3. Blocked pivot
ax = axes[2]
floor(ax, -.2, 4.2)
square(ax, 1, 0, PURPLE)
square(ax, 1, 1, ORANGE)
square(ax, 2, 1, RED)
square(ax, 2, 0, RED, dashed=True)
ax.plot([1.9, 2.95], [.1, .95], color=RED, lw=2)
panel(ax, "Not allowed: corner full", "A pivot is only allowed if the\ncorner cell it swings through\nis empty.")

# 4. Disconnect
ax = axes[3]
floor(ax, -.2, 4.2)
square(ax, 0, 0, PURPLE)
square(ax, 0, 1, GREY)
square(ax, 1, 1, ORANGE)
square(ax, 2, 1, GREY)
ax.add_patch(Rectangle((2.02, 1.02), .96, .96, facecolor="none", edgecolor=RED, lw=2))
square(ax, 1, 0, RED, dashed=True)
ax.plot([.5, .5, 2.5], [.5, 1.5, 1.5], color=TEAL, lw=2.5)
arrow(ax, (1.5, 1.4), (1.5, .55), colour=RED)
ax.text(2.0, 2.15, "would fall", fontsize=8.5, color=RED)
panel(ax, "Not allowed: something falls", "A bot may not move if that would\nleave any bot without a chain of\nlinks (teal) to the floor.")

fig.tight_layout()
fig.savefig("media/linking_moves.png", dpi=130, facecolor="white")
print("saved media/linking_moves.png")
