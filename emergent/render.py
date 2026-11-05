"""Render a run (mid-training or finished) from its checkpoint: samples.png + traj.gif in runs/<name>/peek/.
    python render.py runs/<name> [--mix 1.0]
"""
import os, sys, json, torch
import train
from sim import Physics
from policy import Policy
run = sys.argv[1].rstrip("/")
dev = "cuda"
cfg = dict(train.DEF)
cfg.update(json.loads(open(os.path.join(run, "log.txt")).readline()))
cfg["compile"] = False
ck = os.path.join(run, "ckpt.pt") if os.path.exists(os.path.join(run, "ckpt.pt")) else os.path.join(run, "ema.pt")
s = torch.load(ck, map_location=dev)
data = train.Data(cfg["data"], dev)
ph = Physics(g=1.0, f_bond=cfg["f_bond"], f_act=cfg["f_act"], mu=cfg["mu"],
             reaction=cfg["reaction"], W=data.W, k_c=cfg["k_c"], r_bond=cfg["r_bond"], r_hold=cfg["r_hold"], r_touch=cfg["r_touch"], ctrl=cfg.get("ctrl", "force"), kp=cfg.get("kp", 10.0), air=cfg.get("air", 0.05),
             k_b=cfg.get("k_b", 200.0), dt=cfg.get("dt", 0.02), c_c=cfg.get("c_c", 3.0), c_b=cfg.get("c_b", 2.0))
pol = Policy(H=cfg["H"], M=cfg["M"], rounds=cfg["rounds"], r_sense=cfg["r_sense"], f_act=cfg["f_act"], use_h=cfg["use_h"],
             see_v=cfg["see_v"], grip_bias=cfg["grip_bias"],
             out_scale=cfg.get("v_max", 2.0) if cfg.get("ctrl") == "velocity" else None).to(dev)
pol.load_state_dict(s["ema"])
out = os.path.join(run, "peek"); os.makedirs(out, exist_ok=True)
cfg["name"] = os.path.basename(run) + f" (it {s.get('it', 'final')})"
train.draw(pol, ph, data, cfg, os.path.join(out, "samples.png"), os.path.join(out, "traj.gif"), dev)
print("wrote", out)
