"""Minimal play page for the newest, best swarm models (composed recipe "v3": centred pile, 16x16 map, noisy clock,
repair batches, geometric losses weight 100).   python demo/play.py [--port 8001]   (binds to 127.0.0.1)
Pick a model, pick or draw a shape, watch the swarm form, click to cut a chunk, press Repair.
Weights come from the Hugging Face repo on first use (cached in diffusion/weights_cache/, git-ignored)."""
import argparse, glob, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import engine as E

HERE = E.HERE
LOCK = threading.Lock()
LIB = E.shape_library()
SHAPES = {n: v for n, v in LIB.items() if n.startswith("emoji ") or n.startswith("letter ")}

# Newest family (s14-s19), best seed per sensing range on the common 48-shape test set (demo/lb, tick 58).
MODELS = [
    dict(run="s14_refv3_r40_g16", name="Wide", sees=6.6,
         note="Sees about 6.6 bot spacings. Highest overlap of the newest models."),
    dict(run="s14_refv3_r25_g16", name="Medium", sees=4.1,
         note="Sees about 4.1 bot spacings. Same recipe, shorter range."),
    dict(run="s19_refv3_r14_dmg08_big", name="Arm reach", sees=2.3,
         note="Sees only as far as a real bot's arm reaches (2.3 spacings). Trained on big cuts, so it repairs best."),
    dict(run="s16_refv3_r10_g16", name="Short", sees=1.6,
         note="Sees 1.6 spacings, less than the arm reach. Thin parts soften."),
]
BY = {m["run"]: m for m in MODELS}


def scores():
    lb = {}
    for p in glob.glob(os.path.join(HERE, "lb", "shard_*.json")):
        lb.update(json.load(open(p)))
    return {k: round(v["agg"]["all"], 3) for k, v in lb.items() if k in BY or k == "_rule"}


def pts(a):
    return np.round(np.asarray(a), 3).tolist()


def api_run(req):
    name = req["model"]
    if name not in BY:
        raise ValueError("unknown model")
    occ, sdf = E.maps_from_mask(req["mask"]) if "mask" in req else SHAPES[req["shape"]]
    n = int(np.clip(int(req.get("n", 256)), 64, 1024))
    kw = dict(seed=int(req.get("seed", 0)), K=64, keep_spacings=True, n_ref=n)
    if "x0" in req:                       # repair: survivors restart with the clock at 0
        kw.update(x0=req["x0"], t0=0.0, N=len(req["x0"]))
    else:
        kw.update(N=n)
    r = E.run_model(name, occ, sdf, **kw)
    return dict(frames=[pts(f) for f in r["frames"]], sp=round(r["sp"], 4), occ=np.round(occ).astype(int).tolist(),
                iou=round(r["iou"], 3), cover=round(r["cover"], 3), coll=round(r["coll"], 4), n=len(r["final"]))


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._send(open(os.path.join(HERE, "play.html"), "rb").read(), "text/html; charset=utf-8")
        elif p == "/api/models":
            s = scores()
            self._send(json.dumps(dict(models=[dict(m, iou=s.get(m["run"])) for m in MODELS], rule=s.get("_rule"))))
        elif p == "/api/shapes":   # each shape as a 4096-char '0'/'1' string, row 0 = bottom
            self._send(json.dumps({k: "".join("1" if v > 0.5 else "0" for v in o.reshape(-1)) for k, (o, _) in SHAPES.items()}))
        else:
            self._send("not found", "text/plain", 404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                self._send(json.dumps(api_run(req)))
        except Exception as e:
            self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}), code=400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001); a = ap.parse_args()
    for m in MODELS:
        E.get_model(m["run"])
    print(f"play on http://127.0.0.1:{a.port} | {len(MODELS)} models | {len(SHAPES)} shapes", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
