"""Draw a particle swarm to a PNG with Pillow (for the results figures and quick checks)."""

import numpy as np
from PIL import Image, ImageDraw

import npa

BACKGROUND = (18, 20, 28)


def draw(x, s, size=384, view=1.0, radius=None, mark=None):
    """Render particles as filled discs coloured by their state. view = half-width in world units."""
    x, s = np.asarray(x), np.asarray(s)
    scale = size / (2.0 * view)
    if radius is None:
        radius = max(1.2, 0.5 * scale * 2.0 * view / np.sqrt(len(x)) * 1.5)
    img = Image.new("RGB", (size, size), BACKGROUND)
    d = ImageDraw.Draw(img)
    rgb = (np.clip(np.asarray(npa.colour(s)), 0.0, 1.0) * 255).astype(np.uint8)
    px = (x[:, 0] + view) * scale
    py = (view - x[:, 1]) * scale
    for (a, b), c in zip(zip(px, py), rgb):
        d.ellipse([a - radius, b - radius, a + radius, b + radius], fill=tuple(int(v) for v in c))
    if mark is not None:   # a ring showing where damage was applied: (cx, cy, r) in world units
        cx, cy, r = mark
        a, b, rr = (cx + view) * scale, (view - cy) * scale, r * scale
        d.ellipse([a - rr, b - rr, a + rr, b + rr], outline=(255, 255, 255), width=1)
    return img


def strip(images, gap=4):
    """Put images side by side."""
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, max(i.height for i in images)), BACKGROUND)
    left = 0
    for i in images:
        out.paste(i, (left, 0))
        left += i.width + gap
    return out
