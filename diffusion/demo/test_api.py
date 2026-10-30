"""Smoke test of the running demo server (run on the same box)."""
import json, time, urllib.request
import numpy as np
def post(d):
    r = urllib.request.Request("http://localhost:8000/api/run", data=json.dumps(d).encode(), method="POST")
    try:
        return json.load(urllib.request.urlopen(r, timeout=120))
    except Exception as e:
        return {"error": str(e)}
def show(tag, j, t):
    if "error" in j:
        print(f"{tag:56s} ERROR {j['error']}"); return
    print(f"{tag:56s} {t:4.2f}s inside {j['inside']:.2f} cover {j['cover']:.2f} coll {j['coll']:.3f} frames {len(j['frames'])}")
for model, start in [("s4_ref_local_r60_g8_tn15_dmg_pfc", "pile_c"), ("s4_ref_global_g8_tn15_dmg_pfc", "pile_c"), ("s2_cond_global", "noise"),
                     ("s2_cond_local_r25_L4", "noise"), ("s1_cond_local_r40_L4_notime", "noise"), ("s2pf_cond_global_pf", "pile_off"),
                     ("s1_uncond_global_60k", "noise")]:
    t = time.time(); j = post(dict(model=model, shape="star", start=start, k=16)); show(model + " star " + start, j, time.time() - t)
m = np.zeros((64, 64)); m[40:50, 10:54] = 1; m[10:40, 26:38] = 1            # a drawn "T" (row 0 = bottom)
t = time.time(); j = post(dict(model="s4_ref_local_r60_g8_tn15_dmg_pfc", mask=m.tolist(), start="pile_c", k=16)); show("drawn T, reference swarm", j, time.time() - t)
fin = np.array(j["frames"][-1]); c = fin[np.argmin(np.abs(fin[:, 0]) + np.abs(fin[:, 1] - 0.4))]
keep = fin[np.linalg.norm(fin - c, axis=1) > 0.3]
t = time.time(); j2 = post(dict(model="s4_ref_local_r60_g8_tn15_dmg_pfc", mask=m.tolist(), x0=keep.tolist(), t0=0.0, n=len(keep), k=16)); show(f"repair after cutting {len(fin) - len(keep)} bots", j2, time.time() - t)
t = time.time(); j3 = post(dict(model="s2_cond_global", shape="ring", start="noise", k=16, n=512, hardcore=True, jitter=0.2, motion=0.3)); show("512 bots, jitter, motion noise, hard-core", j3, time.time() - t)
