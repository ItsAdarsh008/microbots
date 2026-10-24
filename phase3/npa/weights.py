"""Load and save NPA weights: NumPy .npz files, the authors' PyTorch .pth, and the web demo's .js.

The authors' lizard.pth is a PyTorch zip archive. Unpickling someone else's file runs
their code, so this reads the raw tensor bytes straight out of the zip instead. It needs
neither PyTorch nor pickle, and it checks every shape and size it expects.
"""

import json
import re
import zipfile
from pathlib import Path

import numpy as np

import npa

# Storage order in the authors' state_dict: eps0, alpha, model.0.weight, model.0.bias, model.2.weight.
_PTH_SHAPES = [(1,), (1,), (npa.HIDDEN_DIMS, npa.PERCEPTION_DIMS), (npa.HIDDEN_DIMS,),
               (npa.OUTPUT_DIMS, npa.HIDDEN_DIMS)]


def load_pth(path):
    """Read the authors' checkpoint. Returns (params, eps0, alpha) with params as NumPy arrays."""
    with zipfile.ZipFile(path) as z:
        names = {}
        for name in z.namelist():
            m = re.fullmatch(r"[^/]+/data/(\d+)", name)
            if m:
                names[int(m.group(1))] = name
        if sorted(names) != list(range(len(_PTH_SHAPES))):
            raise ValueError(f"unexpected tensors in {path}: {sorted(names)}")
        tensors = []
        for i, shape in enumerate(_PTH_SHAPES):
            a = np.frombuffer(z.read(names[i]), dtype="<f4")
            if a.size != int(np.prod(shape)):
                raise ValueError(f"tensor {i} has {a.size} values, expected {int(np.prod(shape))}")
            tensors.append(a.reshape(shape))
    eps0, alpha, w1, b1, w2 = tensors
    if abs(alpha[0] - npa.ALPHA) > 1e-6:
        raise ValueError(f"checkpoint alpha is {alpha[0]}, this code uses {npa.ALPHA}")
    # PyTorch stores Linear weights as (out, in); this code multiplies z @ w, so transpose.
    params = {"w1": w1.T.copy(), "b1": b1.copy(), "w2": w2.T.copy()}
    return params, float(eps0[0]), float(alpha[0])


def save_npz(path, params, eps, eps0):
    np.savez(path, eps=eps, eps0=eps0, **{k: np.asarray(v) for k, v in params.items()})


def load_npz(path):
    """Returns (params, eps, eps0)."""
    d = np.load(path)
    eps = float(d["eps"])
    return {k: d[k] for k in ("w1", "b1", "w2")}, eps, float(d["eps0"]) if "eps0" in d else eps


def to_js(path, name, params, eps, eps0, label="", note="", particles=None, scales=None):
    """Write a <script>-loadable file that registers the model as window.NPA_MODELS[name].

    Plain script (not fetch + JSON) so the demo also works when opened from disk.
    """
    def flat(a):
        return [float(f"{v:.6g}") for v in np.asarray(a, dtype=np.float32).ravel()]

    model = {
        "name": name, "label": label or name, "note": note, "particles": particles, "eps": eps, "eps0": eps0, "alpha": npa.ALPHA,
        "stateDims": npa.STATE_DIMS, "hiddenDims": npa.HIDDEN_DIMS,
        "w1": flat(params["w1"]),   # (66 * len(scales), 128), row-major
        "b1": flat(params["b1"]),   # (128,)
        "w2": flat(params["w2"]),   # (128, 18), row-major
    }
    if scales:
        model["scales"] = [float(v) for v in scales]    # sensing radii as multiples of eps (a model trained with cells.Cell(scales=...))
    text = ("window.NPA_MODELS = window.NPA_MODELS || {};\n"
            f"window.NPA_MODELS[{json.dumps(name)}] = {json.dumps(model, separators=(',', ':'))};\n")
    Path(path).write_text(text)
    return len(text)


def to_js_many(path, models):
    """Write several models into one <script> file. models: list of dicts with the keyword arguments of to_js
    (name, params, eps, eps0, label, note, particles)."""
    parts = ["window.NPA_MODELS = window.NPA_MODELS || {};\n"]
    for m in models:
        tmp = Path(path).with_suffix(".tmp.js")
        to_js(tmp, m["name"], m["params"], m["eps"], m["eps0"], label=m.get("label", ""), note=m.get("note", ""),
              particles=m.get("particles"), scales=m.get("scales"))
        parts.append(tmp.read_text().split("\n", 1)[1])
        tmp.unlink()
    text = "".join(parts)
    Path(path).write_text(text)
    return len(text)
