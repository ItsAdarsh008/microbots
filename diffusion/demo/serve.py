"""Interactive demo server: pick a trained swarm model, pick or draw a shape, watch it form, cut a chunk, repair.
CPU inference (the models are 2.5-3.6M parameters; one run takes well under a second).
    python demo/serve.py --port 8000       (binds to 127.0.0.1; reach it through an SSH tunnel)"""
import argparse, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evalm, shapes
from model import build, make_maps

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("DEMO_DEVICE", "cuda") == "cuda" else "cpu"
DESC = json.load(open(os.path.join(HERE, "models.json")))
MODELS, LOCK = {}, threading.Lock()
for f in sorted(os.listdir(os.path.join(ROOT, "demo_models"))):
    if f.endswith(".pt"):
        n = f[:-3]; blob = torch.load(os.path.join(ROOT, "demo_models", f), map_location="cpu"); cfg = blob["config"]
        m = build(cfg); m.load_state_dict(blob["ema"]); m.to(DEV).eval(); MODELS[n] = (m, cfg)
PRE = {}
for fn in ("shapes2", "shapes"):
    p = os.path.join(ROOT, "data", f"{fn}.npz")
    if os.path.exists(p):
        D = np.load(p)
        for i, nm in enumerate(D["em_names"]):
            PRE.setdefault(str(nm), (D["em_occ"][i][::-1].copy(), D["em_sdf"][i][::-1].copy()))   # emoji images are stored upside-down: flip for display
        for i in range(6):
            PRE.setdefault(f"held-out {fn} #{i}", (D["ood_occ"][i], D["ood_sdf"][i]))
        break
LET = os.path.join(ROOT, "data", "letters.npz")
if os.path.exists(LET):
    Lt = np.load(LET)
    for i, nm in enumerate(Lt["names"]):
        ch, tag = str(nm).split("-")
        PRE[f"letter {ch} ({'thick' if tag == 'bold' else 'thin'})"] = (Lt["occ"][i], Lt["sdf"][i])
R = 64


def maps_from_mask(mask64):
    """User-drawn 64x64 mask (row 0 = bottom, y up) -> occ, sdf in the training format (centred, scaled)."""
    m = np.kron(np.asarray(mask64, bool).reshape(R, R), np.ones((2, 2), bool))
    m = shapes.normalise(m, rad=0.85)
    if m is None:
        raise ValueError("shape too small, too big or empty")
    sdf = shapes.signed_distance(m)
    return m.reshape(R, 2, R, 2).mean((1, 3)).astype(np.float32), sdf[1::2, 1::2][:R, :R].astype(np.float32)


def separate(x, dmin, iters=3):
    for _ in range(iters):
        d = x[:, :, None] - x[:, None]
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x + (torch.relu(dmin - dist) / 2 * d / dist * (dist > 1e-5)).sum(2)
    return x


def run(req):
    m, cfg = MODELS[req["model"]]
    N, K = int(req.get("n", 256)), int(req.get("k", 16))
    rng = np.random.default_rng(int(req.get("seed", 0)))
    torch.manual_seed(int(req.get("seed", 0)))
    if "mask" in req:
        occ, sdf = maps_from_mask(req["mask"])
    else:
        occ, sdf = PRE[req["shape"]]
    maps = make_maps(torch.tensor(occ)[None].to(DEV), torch.tensor(sdf)[None].to(DEV), cfg.get("coarse", 0)) if cfg["cond"] == "map" else None
    sp = evalm.spacing(occ, N)
    ys, xs = np.nonzero(occ > 0.5); ax = (np.arange(R) + 0.5) * 2 / R - 1
    cen = np.array([ax[xs].mean(), ax[ys].mean()])
    start, frames, travel = req.get("start", "noise"), [], 0
    if "x0" in req:
        x = torch.tensor(req["x0"], dtype=torch.float32)[None].to(DEV); t0 = float(req.get("t0", 0.0))
    else:
        t0 = 0.0
        if start == "noise":
            x = torch.tensor(rng.normal(size=(1, N, 2)) * cfg["sigma"], dtype=torch.float32).to(DEV)
        else:
            off = 0.5 if start == "pile_off" else 0.0
            x = torch.tensor(evalm.pile_fps(rng, 1, N, R0=0.7, off=off)).to(DEV)
            if start == "pile_off":      # phase 1: the pile travels as a rigid body to the shape's centre
                shift = torch.tensor(cen, dtype=torch.float32, device=DEV) - x[0].mean(0)
                travel = 10
                for j in range(travel):
                    frames.append((x[0] + shift * j / travel).tolist())
                x = x + shift
    frames.append(x[0].tolist())
    jit, mot, hard, vmax = float(req.get("jitter", 0)), float(req.get("motion", 0)), bool(req.get("hardcore", False)), req.get("vmax")
    with torch.no_grad():
        for i in range(K):
            t = torch.full((1, N), t0 + (1 - t0) * i / K, device=DEV)
            if jit:
                t = (t + torch.randn_like(t) * jit).clamp(0, 1)
            v = m(x, t if jit else t[:, 0], maps)
            dx = v * (1 - t0) / K
            if vmax:
                dx = dx * torch.clamp(float(vmax) * sp / dx.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
            x = x + dx
            if mot:
                x = x + torch.randn_like(x) * mot * sp / np.sqrt(K)
            if hard:
                x = separate(x, 0.7 * sp)
            frames.append(x[0].tolist())
    X = x[0].cpu().numpy()
    mt = evalm.metrics(X, X, occ, sdf)
    return dict(frames=[[[round(a, 3), round(b, 3)] for a, b in f] for f in frames], travel=travel, sp=sp,
                inside=mt["inside"], cover=mt["cover"], coll=mt["coll"], n=int(len(X)),
                occ=np.round(occ, 2).tolist())


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self._send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
        elif self.path == "/api/models":
            self._send(json.dumps([dict(name=n, **DESC.get(n, {}), cfg={k: cfg.get(k) for k in ("radius", "layers", "cond", "time", "start", "coarse", "damage_p", "tnoise", "gstate")})
                                   for n, (m, cfg) in MODELS.items() if n in DESC]))
        elif self.path == "/api/shapes":
            self._send(json.dumps({n: np.round(o, 1).tolist() for n, (o, s) in PRE.items()}))
        else:
            self._send("not found", "text/plain", 404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                self._send(json.dumps(run(req)))
        except Exception as e:
            self._send(json.dumps({"error": str(e)}), code=400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); a = ap.parse_args()
    print("device:", DEV, "| models:", list(MODELS), "| shapes:", len(PRE), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
