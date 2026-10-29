"""Held-out-family errors one family at a time.   python perfam.py runs/<name>"""
import json, sys
import numpy as np, torch
from model import Denoiser
import evalm
from evalm import metrics, agg
run = sys.argv[1]
cfg = json.load(open(f"{run}/result.json"))["config"]
evalm.COARSE = cfg.get("coarse", 0)
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
fams = list(D["families"])
m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"]).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
idx = np.arange(len(D["ood_pts"]))
g = torch.Generator().manual_seed(7)
x0 = evalm.init_noise((len(idx), 256, 2), g, cfg)
maps = evalm.make_maps(torch.tensor(D["ood_occ"]).cuda(), torch.tensor(D["ood_sdf"]).cuda())
X = torch.cat([evalm.sample(m, x0[i:i+64], maps[i:i+64], 32) for i in range(0, len(idx), 64)]).cpu().numpy()
out = []
for f in sorted(set(D["ood_fam"].tolist())):
    s = idx[D["ood_fam"] == f]
    r = agg([metrics(X[i], D["ood_pts"][i], D["ood_occ"][i], D["ood_sdf"][i]) for i in s])
    out.append(f"{fams[f]} (n={len(s)}): w2 {r['w2']:.2f} inside {r['inside']:.2f} cover {r['cover']:.2f} coll {r['coll']:.3f}")
print(run, "|", " | ".join(out))
