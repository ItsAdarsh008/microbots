"""Is the no-learning rule just under-run?  IoU of the rule after more steps, from scattered noise."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iou_eval_lib as L, evalm
from model import make_maps
D = L.D
sets = {"emoji": [(D["em_occ"][i], D["em_sdf"][i]) for i in range(12)] * 2, "held-out": [(D["ood_occ"][i], D["ood_sdf"][i]) for i in range(48)]}
for steps in (200, 600, 1500, 3000):
    r = f"rule {steps:5d} steps:"
    for sn, S in sets.items():
        sc = []
        for i, (o, s) in enumerate(S):
            full = make_maps(torch.tensor(o)[None].cuda(), torch.tensor(s)[None].cuda()); g = torch.Generator().manual_seed(i // 12)
            x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
            sc.append(L.score(evalm.classical(x0, full, torch.tensor([evalm.spacing(o)]).cuda(), steps=steps)[0].cpu().numpy(), o))
        r += f"   {sn} IoU {np.mean([a[0] for a in sc]):.2f} rec {np.mean([a[2] for a in sc]):.2f}"
    print(r, flush=True)
