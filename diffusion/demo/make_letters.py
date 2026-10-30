"""Letters A-Z in two fonts as a test set in the training-shape format (centred, scaled, 256 evenly spaced points).
Out-of-distribution on purpose: nobody trained on letters.   python demo/make_letters.py  (run in diffusion/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shapes
FONTS = {"bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "thin": "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"}
rng = np.random.default_rng(0)
names, P, O, S = [], [], [], []
for tag, path in FONTS.items():
    f = ImageFont.truetype(path, 110)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        im = Image.new("L", (128, 128), 0); d = ImageDraw.Draw(im); w, h = d.textsize(ch, font=f)
        d.text(((128 - w) / 2, (128 - h) / 2 - 8), ch, fill=255, font=f)
        m = shapes.normalise(np.flipud(np.asarray(im) > 127), rad=0.9)   # array row 0 is the bottom (y up); image row 0 is the top
        if m is None:
            continue
        pts, occ, sdf = shapes.pack(m, rng); names.append(f"{ch}-{tag}"); P.append(pts); O.append(occ); S.append(sdf)
np.savez_compressed("data/letters.npz", pts=np.stack(P), occ=np.stack(O), sdf=np.stack(S), names=np.array(names))
print(len(names), "letters")
