"""Run on a GPU box: pack the EMA weights and config of the named runs into demo_export/ (small files).
    python demo/export.py name1 name2 ..."""
import json, os, sys, torch
out = os.path.expanduser("~/diffusion/demo_export"); os.makedirs(out, exist_ok=True)
for n in sys.argv[1:]:
    ck, rj = os.path.expanduser(f"~/diffusion/runs/{n}/ckpt.pt"), os.path.expanduser(f"~/diffusion/runs/{n}/result.json")
    if not (os.path.exists(ck) and os.path.exists(rj)):
        continue
    cfg = json.load(open(rj))["config"]
    torch.save({"ema": torch.load(ck, map_location="cpu")["ema"], "config": cfg}, f"{out}/{n}.pt")
    print("exported", n, round(os.path.getsize(f"{out}/{n}.pt") / 1e6, 1), "MB")
