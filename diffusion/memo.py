"""Memorisation check for an unconditional run: best raster overlap (IoU) of generated shapes with the shapes the
model was TRAINED on (first n_train), versus the same number for real validation shapes (never seen).
Copying shows as generated >> real.    python memo.py runs/<name>"""
import json, sys
import numpy as np, torch
from model import Denoiser
import evalm
run = sys.argv[1]
cfg = json.load(open(f"{run}/result.json"))["config"]
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
n = cfg["n_train"]
mt = torch.tensor(D["tr_occ"][:n].reshape(n, -1) > 0.5).cuda().float()
m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"]).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
g = torch.Generator().manual_seed(5)
x0 = (torch.randn((256, 256, 2), generator=g) * cfg["sigma"]).cuda()
X = torch.cat([evalm.sample(m, x0[i:i+64], None, 32) for i in range(0, 256, 64)]).cpu().numpy()
def best(Xs):
    M = torch.tensor(np.stack([evalm.raster(x) for x in Xs]).reshape(len(Xs), -1)).cuda().float()
    inter = M @ mt.T
    iou = inter / (M.sum(1, keepdim=True) + mt.sum(1)[None] - inter)
    return iou.max(1).values.cpu().numpy()
gen, real = best(X), best(D["va_pts"][:256])
print(f"{run}: trained on {n} shapes. best IoU with a TRAINING shape: generated {gen.mean():.3f}  real unseen {real.mean():.3f}  "
      f"(frac generated >0.9: {np.mean(gen>0.9):.2f}, real: {np.mean(real>0.9):.2f})")
