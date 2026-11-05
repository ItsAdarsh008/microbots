"""Render every lattice run on the SAME six shapes (tower, gamma, bridge, tee, mushroom, tree) as an animated GIF plus
a final-frame PNG, and write its settings and latest metrics to JSON, so runs can be compared side by side.

    python lat/gallery.py runs/lat1_L1 runs/lat2_C0 ...     # also: @untrained, @perfect (references at L = 1)
Output: gallery/<name>.gif, gallery/<name>.png, gallery/<name>.json

Colours: blue = target cell filled, dark blue = cured target bot, orange = bot outside the target, brown = cured
bot outside the target, red = unheld (falling) bot, light grey = target cell still empty.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es  # noqa: E402
from latworld import support_span, gravity  # noqa: E402

SHOW = ["tower", "gamma", "bridge", "tee", "mushroom", "tree"]
COL = dict(bg=(248, 247, 243), empty_t=(214, 214, 206), fill=(47, 84, 150), fill_c=(22, 42, 88),
           out=(226, 128, 52), out_c=(140, 72, 20), fall=(208, 44, 44), floor=(60, 60, 60), text=(30, 30, 30),
           panel=(255, 255, 255))
OUT = "gallery"


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


F_HEAD, F_LAB = font(13), font(11)


def shapes(H, W, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for fam in SHOW:
        while True:
            t = es.gen_target(fam, rng, H, W)
            if t is not None and 12 <= t.sum() <= 110:
                out.append(t); break
    return torch.tensor(np.array(out), dtype=torch.float32)


def run_cfg(d):
    cfg = dict(es.DEF)
    cfg.update(json.loads([l for l in open(os.path.join(d, "log.txt")) if l.startswith("{")][-1]))
    return cfg


def parse_metrics(d):
    lines = open(os.path.join(d, "log.txt")).read().splitlines()
    num = lambda l: {k: float(v) for k, v in re.findall(r"([a-z_]+) (-?[0-9.]+)", l.split(":", 1)[1])}
    res = {}
    fin = {l.split(":")[0].split()[1]: num(l) for l in lines if l.startswith("final ")}
    if fin:
        res["final"] = fin
    ev = [l for l in lines if l.startswith("eval gen")]
    if ev:
        res["last_eval_gen"] = int(ev[-1].split()[2].rstrip(":"))
        res["last_eval"] = num(ev[-1])
    gl = [l for l in lines if l.startswith("gen ")]
    res["gen"] = int(gl[-1].split()[1]) if gl else 0
    return res


def status(d, name):
    if os.path.exists(os.path.join(d, "result.json")):
        return "done"
    r = subprocess.run(["pgrep", "-f", f"[l]at/es.py.*{name}\\.json"], capture_output=True)
    return "running" if r.returncode == 0 else "stopped"


def render(xframes, tgt0, cfg, label, path):
    E, H, W = tgt0.shape
    c = 5 if W <= 40 else 3
    pw, ph = W * c, H * c
    gap, head, lab = 6, 22, 15
    cols, rows = 3, 2
    Wimg = cols * pw + (cols + 1) * gap
    Himg = head + rows * (ph + lab + 3) + (rows + 1) * gap
    T = cfg["T"]
    build = [x for x in xframes if x[0] == "build"]
    rest = [x for x in xframes if x[0] != "build"]
    k = max(1, len(build) // 70)
    seq = build[::k] + [build[-1]] + rest
    imgs, durs = [], []
    for ph_name, t, snap in seq:
        occ, unheld, cured, tg = [snap[i].numpy() for i in range(4)]
        im = Image.new("RGB", (Wimg, Himg), COL["bg"])
        dr = ImageDraw.Draw(im)
        step = f"step {t}/{T}" if ph_name == "build" else f"{ph_name} (no actions)"
        dr.text((gap, 4), label, fill=COL["text"], font=F_HEAD)
        tw = dr.textlength(step, font=F_LAB)
        dr.text((Wimg - gap - tw, 6), step, fill=COL["text"], font=F_LAB)
        for e in range(min(E, 6)):
            r_, c_ = divmod(e, cols)
            x0 = gap + c_ * (pw + gap)
            y0 = head + gap + r_ * (ph + lab + 3 + gap)
            nt = tg[e].sum()
            fill = (occ[e] * tg[e]).sum() / max(nt, 1)
            txt = f"{SHOW[e]}" + (f"  fill {fill:.2f}" if cfg['obj'] == 'fill' else "")
            dr.text((x0, y0), txt, fill=COL["text"], font=F_LAB)
            py = y0 + lab
            arr = np.empty((H, W, 3), np.uint8)
            arr[:] = COL["panel"]
            t_, o_, u_, cu = tg[e] > 0, occ[e] > 0, unheld[e] > 0, cured[e] > 0
            if cfg["obj"] == "fill":
                arr[t_ & ~o_] = COL["empty_t"]
                arr[t_ & o_] = COL["fill"]
                arr[t_ & o_ & cu] = COL["fill_c"]
                arr[~t_ & o_] = COL["out"]
                arr[~t_ & o_ & cu] = COL["out_c"]
            else:
                arr[o_] = COL["out"]
                arr[o_ & cu] = COL["out_c"]
            arr[u_] = COL["fall"]
            arr = arr[::-1]                                    # row 0 = floor at the bottom
            pim = Image.fromarray(arr).resize((pw, ph), Image.NEAREST)
            im.paste(pim, (x0, py))
            dr.line([(x0, py + ph), (x0 + pw - 1, py + ph)], fill=COL["floor"], width=2)
        imgs.append(im.quantize(colors=32, method=Image.MEDIANCUT, dither=Image.NONE))
        durs.append(60)
    durs[-1] = 2500
    if len(build) and seq:
        # also pause on the last build frame
        durs[len(build[::k])] = 700
    imgs[0].save(path + ".gif", save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    imgs[-1].convert("RGB").save(path + ".png")


def main(args):
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in args:
        if d.startswith("@"):
            name = d[1:]
            cfg = dict(es.DEF)
            info = dict(name=name, status="reference", cfg=cfg, metrics={})
        else:
            name = os.path.basename(d.rstrip("/"))
            ck = os.path.join(d, "theta.pt") if os.path.exists(os.path.join(d, "theta.pt")) else os.path.join(d, "ckpt.pt")
            if not os.path.exists(ck):
                print(name, "no checkpoint"); continue
            cfg = run_cfg(d)
            info = dict(name=name, status=status(d, name), cfg=cfg, metrics=parse_metrics(d))
        H, W = cfg["H"], cfg["W"]
        tg = shapes(H, W).to(dev)
        nca = es.make_nca(cfg)
        g = torch.Generator(device=dev).manual_seed(7)
        if name == "perfect":
            # the target placed perfectly, then the same settling physics (what the target alone keeps)
            occ = tg.clone(); hid = torch.zeros(6, cfg["C"], H, W, device=dev)
            s = support_span(occ, cfg["L"], None, H + W)
            xf = [("build", cfg["T"], es._snap(occ, s, None, torch.zeros_like(occ), 0, tg, cfg["L"]))]
            for k in range(cfg["settle"]):
                s = support_span(occ, cfg["L"], s, cfg["support_iters"])
                occ, hid, _ = gravity(occ, hid, cfg["L"], s)
                if k % 3 == 0:
                    xf.append(("settle", k, es._snap(occ, s, None, torch.zeros_like(occ), 0, tg, cfg["L"])))
            fill = (occ * tg).flatten(1).sum(1) / tg.flatten(1).sum(1)
        else:
            theta = nca.init(torch.Generator().manual_seed(0)).to(dev) if name == "untrained" else \
                torch.load(ck, map_location=dev)["theta"]
            with torch.no_grad():
                _, st = es.rollout(nca, theta[None], tg, cfg, g, record=True,
                                   tgt2=tg.roll(1, 0) if cfg["switch"] > 0 else None)
            xf, fill = st["xframes"], st["fill"]
        info["show_fill"] = {f: float(v) for f, v in zip(SHOW, fill)}
        lab = {"untrained": "untrained controller (uniform random moves), L = 1",
               "perfect": "reference: target placed perfectly, then settled, L = 1"}.get(name, name)
        render(xf, tg.cpu(), cfg, lab, os.path.join(OUT, name))
        json.dump(info, open(os.path.join(OUT, name + ".json"), "w"), indent=1, default=str)
        print(name, info["status"], " ".join(f"{v:.2f}" for v in info["show_fill"].values()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
