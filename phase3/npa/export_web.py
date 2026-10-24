"""Build the web demo's data files: the target outline and the model weights.

    python phase3/npa/export_web.py --ours models/lizard_1024.npz     # one of our own trained models
    python phase3/npa/export_web.py --fetch-paper                     # the authors' pretrained lizard
    python phase3/npa/export_web.py --paper path/to/lizard.pth        # the same, from a file you have

The authors' weights are not committed to this repo (their repo carries no licence), so the
paper model is built on your machine and stays out of git (web/weights/lizard_paper.js is
ignored). Everything is read as plain numbers; the .pth is never unpickled.
"""

import argparse
import base64
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

import target as tgt
import weights

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
PAPER_URL = "https://raw.githubusercontent.com/TheDevilWillBeBee/NPA/main/data/pretrained/lizard.pth"


def write_target():
    """web/target.js: the lizard's silhouette as packed bits, and its centre, for the outline and match score."""
    t = tgt.load_target()
    side = t["side"]
    mask = np.zeros(side * side, dtype=np.uint8)
    cols = np.round((t["pos"][:, 0] + tgt.AABB) / t["pixel"] - 0.5).astype(int)
    rows = np.round((tgt.AABB - t["pos"][:, 1]) / t["pixel"] - 0.5).astype(int)
    mask[rows * side + cols] = 1
    packed = np.packbits(mask, bitorder="little")
    centroid = t["pos"].mean(axis=0)
    data = {"side": side, "centroid": [float(centroid[0]), float(centroid[1])],
            "mask": base64.b64encode(packed.tobytes()).decode()}
    (WEB / "target.js").write_text(f"window.NPA_TARGET = {json.dumps(data, separators=(',', ':'))};\n")
    print("wrote web/target.js:", side, "x", side, "pixels,", int(mask.sum()), "in the lizard")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", help="an .npz written by train.py")
    ap.add_argument("--paper", help="the authors' lizard.pth")
    ap.add_argument("--fetch-paper", action="store_true", help="download the authors' lizard.pth, then convert it")
    args = ap.parse_args()

    (WEB / "weights").mkdir(exist_ok=True)
    write_target()

    paper = args.paper
    if args.fetch_paper:
        paper = str(Path(tempfile.mkdtemp()) / "lizard.pth")
        print("downloading", PAPER_URL)
        urllib.request.urlretrieve(PAPER_URL, paper)
    if paper:
        params, eps0, _ = weights.load_pth(paper)
        n = weights.to_js(WEB / "weights" / "lizard_paper.js", "lizard_paper", params, eps0, eps0,
                          label="Paper's pretrained lizard (4,096 particles)", particles=4096,
                          note="Weights released by the paper's authors, run through our reimplementation.")
        print("wrote web/weights/lizard_paper.js", n, "bytes")
    if args.ours:
        params, eps, eps0 = weights.load_npz(args.ours)
        n = weights.to_js(WEB / "weights" / "lizard_ours.js", "lizard_ours", params, eps, eps0,
                          label="Our model (trained here)", particles=None,
                          note="Trained with this repo's train.py.")
        print("wrote web/weights/lizard_ours.js", n, "bytes")


if __name__ == "__main__":
    main()
