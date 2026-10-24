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
NAMES = {"all_damage_10k": "Ours: trained to repair every kind of damage (1,024 particles)",
         "base_10k": "Ours: trained like the paper, wipes only (1,024 particles)"}
NOTES = {"all_damage_10k": "Trained here on state wipes, cuts, tears and clumping. NOTE: it was meant to train for 10,000 iterations but "
                           "stopped at about 3,000 after repeated blow-ups, so treat it as a 3,000-iteration model. In our tests it repairs cuts "
                           "and clumps at every size and tears up to 2 sensing radii; a piece torn off 4 radii away is only about half recovered.",
         "base_10k": "Trained here the way the paper does (state wipes only), the full 10,000 iterations. Wiped colour heals; cuts and tears mostly do not."}
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
    ap.add_argument("--ours", nargs="+", help="names of runs in runs/ (plain-network runs with a params.npz)")
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
                          note="Weights released by the paper's authors, run through our reimplementation. Wiped colour and small lost feet heal; a cut tail tip and a piece torn far from the body do not.")
        print("wrote web/weights/lizard_paper.js", n, "bytes")
    if args.ours:
        import json as _json
        models = []
        for run in args.ours:
            d = HERE / "runs" / run
            params, eps, eps0 = weights.load_npz(d / "params.npz")
            info = _json.loads((d / "result.json").read_text())
            cfg, sm = info["config"], info["summary"]
            if cfg["cell"] or cfg.get("train", {}).get("mode", "bptt") != "bptt":
                raise SystemExit(f"{run}: the web engine only runs the plain 66-128-18 network; this run uses {cfg['cell']}")
            models.append({"name": f"ours_{run}", "params": params, "eps": eps, "eps0": eps0, "particles": cfg["particles"],
                           "label": NAMES.get(run, f"Ours: {run}"), "note": NOTES.get(run, f"Trained here ({run}).")})
        n = weights.to_js_many(WEB / "weights" / "ours.js", models)
        print("wrote web/weights/ours.js", n, "bytes:", [m["name"] for m in models])

if __name__ == "__main__":
    main()
