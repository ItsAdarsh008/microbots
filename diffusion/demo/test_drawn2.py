"""Hand-drawn shapes (T, L, disc, ring...) across the given models, started the way each was trained.
   python demo/test_drawn2.py model1 model2 ...   (demo server must be running on localhost:8000)"""
import sys, json, urllib.request
import numpy as np
def post(d):
    r = urllib.request.Request("http://localhost:8000/api/run", data=json.dumps(d).encode(), method="POST")
    return json.load(urllib.request.urlopen(r, timeout=120))
yy, xx = np.mgrid[0:64, 0:64]
def box(a, b, c, d):
    m = np.zeros((64, 64)); m[a:b, c:d] = 1; return m
shapes = {"disc": ((xx - 32) ** 2 + (yy - 32) ** 2 < 22 ** 2).astype(float),
          "fat T": np.maximum(box(34, 48, 8, 56), box(8, 34, 24, 40)), "thin T": np.maximum(box(40, 48, 8, 56), box(8, 40, 28, 36)),
          "L": np.maximum(box(8, 56, 12, 26), box(8, 22, 12, 52)), "plus": np.maximum(box(28, 36, 8, 56), box(8, 56, 28, 36)),
          "ring": (((xx - 32) ** 2 + (yy - 32) ** 2 < 24 ** 2) & ((xx - 32) ** 2 + (yy - 32) ** 2 > 12 ** 2)).astype(float)}
models = sys.argv[1:]
print(f"{'shape':10s}" + "".join(f"{m[:26]:>30s}" for m in models))
for n, m in shapes.items():
    row = f"{n:10s}"
    for mod in models:
        info = [x for x in json.load(urllib.request.urlopen("http://localhost:8000/api/models")) if x["name"] == mod]
        start = (info[0]["start"] if info else "noise")
        j = post(dict(model=mod, mask=m.tolist(), start=start, k=16)); row += f"   inside {j['inside']:.2f} cover {j['cover']:.2f}      "
    print(row)
