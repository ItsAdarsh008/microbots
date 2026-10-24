"""The target shapes: Noto Emoji (Apache-2.0), chosen for their variety of topology.

    star, bone, fish, cactus   no holes (convex, dumbbell, thin fins, branches)
    snowflake, octopus         branching limbs
    butterfly, donut, ring,    one hole, from thick (donut) to thin (ring, key, anchor, wrench)
    key, anchor, wrench
    lizard                     the paper's target
"""

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SHAPES = {"lizard": DATA / "lizard_u1f98e.png"}
for _p in sorted((DATA / "shapes").glob("*.png")):
    SHAPES[_p.stem] = _p

# the suite used for "does the conclusion hold on other shapes?"
SUITE = ("lizard", "star", "bone", "butterfly", "donut", "key", "snowflake", "octopus")


def path(name):
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; choose from {sorted(SHAPES)}")
    return SHAPES[name]
