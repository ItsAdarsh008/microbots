"""Swarm dashboard + live lab for ALL trained runs.   python demo/dashboard.py [--port 8000]   (binds to 127.0.0.1)
Weights are fetched from the Hugging Face repo on first use and cached in diffusion/weights_cache/ (git-ignored)."""
import argparse, glob, json, mimetypes, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import engine as E
HERE, ROOT = E.HERE, E.ROOT
LOCK = threading.Lock()
LIB = E.shape_library()


def leaderboard():
    lb = {}
    for p in glob.glob(os.path.join(HERE, "lb", "shard_*.json")):
        try:
            lb.update(json.load(open(p)))
        except Exception:
            pass
    return lb


def pts(a):
    return np.round(np.asarray(a), 3).tolist()


def pack(r, frames=True):
    d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("final", "frames")}
    d["final"] = pts(r["final"])
    if frames and "frames" in r:
        d["frames"] = [pts(f) for f in r["frames"]]
    return d


def target(req):
    if "mask" in req:
        return E.maps_from_mask(req["mask"])
    return LIB[req["shape"]]


def kw(req):
    return dict(N=int(req.get("n", 256)), K=int(req.get("k", 64)), seed=int(req.get("seed", 0)), start=req.get("start", "auto"),
                jitter=float(req.get("jitter", 0)), motion=float(req.get("motion", 0)), hardcore=bool(req.get("hardcore", False)),
                vmax=req.get("vmax") or None, keep_spacings=bool(req.get("keep_spacings", False)))


def api_run(req):
    occ, sdf = target(req); k = kw(req)
    if "x0" in req:
        k.update(x0=req["x0"], t0=float(req.get("t0", 0.0)), N=len(req["x0"]))
    out = dict(model=pack(E.run_model(req["model"], occ, sdf, **k)), occ=np.round(occ, 2).tolist())
    if req.get("baselines"):
        out["rule"] = pack(E.run_rule(occ, sdf, k["N"], k["seed"]), False)
        out["ceiling"] = pack(E.ceiling(occ, sdf, k["N"], k["seed"]), False)
    return out


def api_compare(req):
    occ, sdf = target(req); k = kw(req); res = {}
    for name in req["models"]:
        res[name] = pack(E.run_model(name, occ, sdf, max_frames=33, **k))
    return dict(models=res, rule=pack(E.run_rule(occ, sdf, k["N"], k["seed"]), False),
                ceiling=pack(E.ceiling(occ, sdf, k["N"], k["seed"]), False), occ=np.round(occ, 2).tolist())


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store" if ctype != "image/png" else "max-age=3600"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._send(open(os.path.join(HERE, "dashboard.html"), "rb").read(), "text/html; charset=utf-8")
        elif p == "/api/catalog":
            self._send(json.dumps(dict(runs=list(E.CATALOG.values()), lb=leaderboard())))
        elif p == "/api/shapes":
            self._send(json.dumps(sorted(LIB)))
        elif p.startswith("/api/shape/"):
            from urllib.parse import unquote
            n = unquote(p[len("/api/shape/"):])
            self._send(json.dumps(np.round(LIB[n][0], 1).tolist())) if n in LIB else self._send("{}", code=404)
        elif p.startswith("/results/"):
            parts = p.split("/")
            if len(parts) == 4 and parts[2] in E.CATALOG and parts[3] in ("samples.png", "traj.gif", "log.txt"):
                f = os.path.join(ROOT, "results", parts[2], parts[3])
                if os.path.exists(f):
                    return self._send(open(f, "rb").read(), mimetypes.guess_type(f)[0] or "text/plain")
            self._send("not found", "text/plain", 404)
        else:
            self._send("not found", "text/plain", 404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                self._send(json.dumps(api_run(req) if self.path == "/api/run" else api_compare(req)))
        except Exception as e:
            self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}), code=400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); a = ap.parse_args()
    print(f"dashboard on http://127.0.0.1:{a.port} | {len(E.CATALOG)} runs | {len(LIB)} shapes", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
