"""Do drawn shapes work?  Compare simple drawings across models (run on the demo box)."""
import json, urllib.request
import numpy as np
def post(d):
    r = urllib.request.Request("http://localhost:8000/api/run", data=json.dumps(d).encode(), method="POST")
    return json.load(urllib.request.urlopen(r, timeout=120))
yy, xx = np.mgrid[0:64, 0:64]
shapes = {}
shapes["disc"] = ((xx - 32) ** 2 + (yy - 32) ** 2 < 22 ** 2).astype(float)
shapes["fat T (bar 14, stem 16 cells)"] = np.zeros((64, 64)); shapes["fat T (bar 14, stem 16 cells)"][34:48, 8:56] = 1; shapes["fat T (bar 14, stem 16 cells)"][8:34, 24:40] = 1
shapes["thin T (bar 8, stem 8)"] = np.zeros((64, 64)); shapes["thin T (bar 8, stem 8)"][40:48, 8:56] = 1; shapes["thin T (bar 8, stem 8)"][8:40, 28:36] = 1
shapes["L"] = np.zeros((64, 64)); shapes["L"][8:56, 12:26] = 1; shapes["L"][8:22, 12:52] = 1
shapes["ring"] = (((xx - 32) ** 2 + (yy - 32) ** 2 < 24 ** 2) & ((xx - 32) ** 2 + (yy - 32) ** 2 > 12 ** 2)).astype(float)
models = [("s4_ref_local_r60_g8_tn15_dmg_pfc", "pile_c"), ("s4_ref_global_g8_tn15_dmg_pfc", "pile_c"), ("s2_cond_global", "noise"), ("s2_cond_local_r60_L4", "noise")]
print(f"{'shape':32s}" + "".join(f"{m[0][:22]:>26s}" for m in models))
for n, m in shapes.items():
    row = f"{n:32s}"
    for mod, st in models:
        j = post(dict(model=mod, mask=m.tolist(), start=st, k=16))
        row += f"   in {j['inside']:.2f} cov {j['cover']:.2f}      "
    print(row)
# the preset star through the SAME mask path, to separate shape effects from path bugs
import urllib.request as u
sh = json.load(u.urlopen("http://localhost:8000/api/shapes"))
for name in ("star", "ring", "donut"):
    if name in sh:
        row = f"{'preset '+name+' via mask path':32s}"
        for mod, st in models:
            j = post(dict(model=mod, mask=(np.array(sh[name]) > 0.5).astype(float).tolist(), start=st, k=16)); row += f"   in {j['inside']:.2f} cov {j['cover']:.2f}      "
        print(row)
