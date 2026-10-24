"""Write experiments into the master queue (lab/queue/master/); fleet.py hands them out to the GPUs.

    python lab/plan.py [--iters 3000]

Skips any name that is already queued, dispatched or finished. Priority prefixes: 05 seed-noise baselines,
10-30 the screening ladder, 35 hypothesis tests (see hypotheses.md), 45 shape sweep. fleet.py's generator
adds more (replicates, mutations, other shapes) when the queue runs low.
"""

import argparse
import copy
import json
from pathlib import Path

import shapes

LAB = Path(__file__).resolve().parent
MASTER = LAB / "queue" / "master"
WIPE = {"kind": "wipe", "level": [0.2, 0.2], "p": 1.0}      # what the paper trains on
CUT = {"kind": "cut", "level": [0.1, 0.3], "p": 0.5}
DRAG = {"kind": "drag", "level": [0.2, 0.8], "p": 0.3}
FAR_DRAG = {"kind": "drag", "level": [0.4, 1.2], "p": 0.4}
CLUMP = {"kind": "clump", "level": [0.3, 0.7], "p": 0.3}
ALL_DAMAGE = [WIPE, CUT, DRAG, CLUMP]
MEM = {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True, "mem": 8}

BASE = {"particles": 1024, "seed": 0, "shape": "lizard", "cell": {},
        "train": {"iters": 3000, "batch": 8, "grid": 128, "damage": [WIPE]}}


def variant(name, cell=None, train=None, shape=None, seed=0):
    cfg = copy.deepcopy(BASE)
    cfg["name"], cfg["seed"] = name, seed
    if shape:
        cfg["shape"] = shape
    cfg["cell"].update(cell or {})
    cfg["train"].update(copy.deepcopy(train or {}))
    return cfg


def experiments():
    out = [(5, variant("base")), (5, variant("base_s1", seed=1)), (5, variant("base_s2", seed=2)),
           (3, variant("base_10k", train={"iters": 10000}))]
    out += [  # capacity ladder (H8)
        (20, variant("width256", {"width": 256})),
        (20, variant("depth3", {"depth": 3})),
        (20, variant("residual3", {"block": "residual", "depth": 3, "zero_head": True})),
        (20, variant("gated3", {"block": "gated", "depth": 3, "zero_head": True})),
        (20, variant("gated3_rms", {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True})),
        (20, variant("gated3_rms_mem8", MEM)),
        # training side (H1, H5)
        (30, variant("noise_state", train={"noise_state": 0.02})),
        (30, variant("noise_both", train={"noise_state": 0.02, "noise_pos": 0.05})),
        (30, variant("wipe_big", train={"damage": [{"kind": "wipe", "level": [0.2, 0.6], "p": 1.0}]})),
        (30, variant("train_cut", train={"damage": [WIPE, CUT]})),
        (30, variant("train_cut_drag", train={"damage": [WIPE, CUT, DRAG]})),
        (30, variant("train_all_damage", train={"damage": ALL_DAMAGE})),
        (30, variant("train_all_damage_noise", train={"damage": ALL_DAMAGE, "noise_state": 0.02, "noise_pos": 0.05})),
        (30, variant("clip1", train={"grad": 1.0})),
        # hypothesis tests (H2, H3, H4, H6)
        (35, variant("train_all_damage_mem8", MEM, {"damage": ALL_DAMAGE})),
        (35, variant("train_far_drag_mem8", MEM, {"damage": [WIPE, CUT, FAR_DRAG]})),
        (35, variant("scales2", {"scales": [1.0, 2.0]})),
        (35, variant("scales2_all_damage", {"scales": [1.0, 2.0]}, {"damage": ALL_DAMAGE})),
        (35, variant("long_rollout", train={"steps": [64, 160]})),
    ]
    # new architecture families and the truncated-gradient (diffusion-style) trainer
    out += [
        (25, variant("edge_default", {"kind": "edge"})),
        (25, variant("edge_radial", {"kind": "edge", "move": "radial"})),
        (25, variant("edge_sph_mem8", {"kind": "edge", "sph": True, "mem": 8})),
        (25, variant("attn_default", {"kind": "attn"})),
        (25, variant("attn_nosph", {"kind": "attn", "sph": False})),
        (25, variant("attn_rms_mem8", {"kind": "attn", "norm": "rms", "mem": 8})),
        (32, variant("edge_all_damage", {"kind": "edge"}, {"damage": ALL_DAMAGE})),
        (32, variant("attn_all_damage", {"kind": "attn"}, {"damage": ALL_DAMAGE})),
        (33, variant("trunc_w8", train={"mode": "trunc", "window": 8})),
        (33, variant("trunc_w8_all_damage", train={"mode": "trunc", "window": 8, "damage": ALL_DAMAGE})),
        (33, variant("trunc_w8_long", train={"mode": "trunc", "window": 8, "steps": [64, 192]})),
        (33, variant("trunc_w4_last", train={"mode": "trunc", "window": 4, "window_loss": "last"})),
    ]
    # first results: deletion-trained models repair cuts (H1) but drift (H17); combine with what grows well and what stabilises
    gat = {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True}
    out += [
        (31, variant("all_damage_long", train={"damage": ALL_DAMAGE, "steps": [64, 160]})),
        (31, variant("gated_all_damage", gat, {"damage": ALL_DAMAGE})),
        (31, variant("gated_trunc_all_damage", gat, {"mode": "trunc", "window": 8, "damage": ALL_DAMAGE})),
        (31, variant("noise_small_all_damage", train={"damage": ALL_DAMAGE, "noise_state": 0.005, "noise_pos": 0.01})),
        (31, variant("gated_all_damage_10k", gat, {"damage": ALL_DAMAGE, "iters": 10000})),
        (31, variant("all_damage_10k", train={"damage": ALL_DAMAGE, "iters": 10000})),
    ]
    # H13: make the memory cell train
    out += [
        (21, variant("gated3_rms_mem8_guard", MEM)),
        (21, variant("mem8_lr3e4", MEM, {"lr": 3e-4})),
        (21, variant("mem8_clip1", MEM, {"grad": 1.0})),
    ]
    # phase 2a: one network, several shapes, against equal-budget single-shape specialists
    ms_shapes = ["lizard", "star", "bone", "butterfly"]
    gated = {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True}

    def multi(name, cell, train=None):
        cfg = variant(name, cell, {"iters": 10000, "mode": "multishape", **(train or {})})
        cfg.pop("shape")
        cfg["shapes"] = ms_shapes
        return cfg

    out += [
        (36, multi("ms4_base", {})),
        (36, multi("ms4_gated3_rms", gated)),
        (37, multi("ms4_gated3_rms_all_damage", gated, {"damage": ALL_DAMAGE})),
    ]
    for shape in ms_shapes:
        out.append((38, variant(f"spec_gated_{shape}_10k", gated, {"iters": 10000}, shape=shape)))
        if shape != "lizard":
            out.append((38, variant(f"spec_base_{shape}_10k", train={"iters": 10000}, shape=shape)))
    # the physical sensing regime (embodiment.md D2): eps = two body diameters, about 13 neighbours instead of 120
    e13 = lambda name, cell=None, train=None: {**variant(name, cell, train), "eps": 0.066}
    wipe13 = {"kind": "wipe", "level": [0.066, 0.066], "p": 1.0}
    all13 = [wipe13] + ALL_DAMAGE[1:]
    out += [
        (28, e13("eps13_base", train={"damage": [wipe13]})),
        (28, e13("eps13_gated3_rms", {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True}, {"damage": [wipe13]})),
        (29, e13("eps13_all_damage", train={"damage": all13})),
        (29, e13("eps13_gated3_rms_all_damage", {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True}, {"damage": all13})),
        (29, e13("eps13_edge", {"kind": "edge", "K": 24}, {"damage": [wipe13]})),
        (29, e13("eps13_attn", {"kind": "attn", "K": 24}, {"damage": [wipe13]})),
    ]
    # phase 2b: shape-agnostic healing. The target is the swarm itself; trained on random + emoji shapes, scored on held-out ones.
    def agn(name, cell, train=None, eps=None):
        cfg = variant(name, cell, {"mode": "agnostic", "iters": 3000, "batch": 4, **(train or {})})
        cfg.pop("shape")
        cfg["train"].pop("damage")          # use the trainer's default: all six damage kinds
        if eps:
            cfg["eps"] = eps
        return cfg

    out += [
        (34, agn("ag_mlp", {})),
        (34, agn("ag_mlp_mem8", {"mem": 8})),
        (34, agn("ag_gated3_rms", gat)),
        (34, agn("ag_gated3_rms_mem8_lr3e4", MEM, {"lr": 3e-4})),
        (34, agn("ag_edge_mem8", {"kind": "edge", "mem": 8})),
        (34, agn("ag_attn_mem8", {"kind": "attn", "mem": 8})),
    ]
    # H14: no shared compass. Every bot gets a random fixed heading; compare the plain MLP, the equivariant cell, and a shared frame.
    hd = lambda name, cell, train=None: variant(name, cell, {"damage": ALL_DAMAGE, **(train or {})})
    out += [
        (27, hd("heading_none_mlp", {"kind": "mlph", "heading": "none"})),
        (27, hd("heading_random_mlp", {"kind": "mlph", "heading": "random"})),
        (27, hd("heading_none_equiv", {"kind": "equiv", "heading": "none"})),
        (27, hd("heading_random_equiv", {"kind": "equiv", "heading": "random"})),
        (27, hd("heading_random_mlp_scales2", {"kind": "mlph", "heading": "random", "scales": [1.0, 2.0]})),
        (27, hd("heading_random_equiv_mem4", {"kind": "equiv", "heading": "random", "mem": 4})),
    ]
    # H15/H16: train INSIDE the physical embodiment (strides, noisy 8-bit sensing, packet loss) at eps 0.05, about 7 neighbours
    PHYS = {"mode": "phys", "sense_noise": 0.0017, "quant_bits": 8, "dropout": 0.3, "body_r": 0.0129, "port_d": 0.0121,
            "max_step": 0.043, "min_step": 0.0155, "step_len_noise": 0.2, "step_dir_noise": 0.087, "bearing_noise": 0.087,
            "stop_proj_grad": True}
    all7 = [{"kind": "wipe", "level": [0.05, 0.05], "p": 1.0}] + ALL_DAMAGE[1:]

    def phys(name, cell):
        cfg = variant(name, {**cell, "embody": PHYS}, {"damage": all7})
        cfg["eps"], cfg["seed_radius"] = 0.05, 0.55
        return cfg

    out += [
        (26, phys("physB_mlp_all_damage_eps7", {})),
        (26, phys("physB_scales2_all_damage_eps7", {"scales": [1.0, 2.0]})),
        (26, phys("physB_gated_all_damage_eps7", gat)),
        (26, phys("physB_equiv_random_eps7", {"kind": "equiv", "heading": "random"})),
    ]
    # H18 fix: rerun the physical-regime configs now that the seed radius is max(2 eps, 0.4)
    for old in ("eps13_base", "eps13_gated3_rms", "eps13_all_damage", "eps13_gated3_rms_all_damage", "eps13_edge", "eps13_attn"):
        for prio, cfg in list(out):
            if cfg["name"] == old:
                v2 = {**cfg, "name": old + "_v2"}
                out.append((prio, v2))
    # H20: shape-agnostic healing with a real budget, and with the cells that fit it (equivariant / random heading)
    out += [
        (24, agn("ag10k_mlp", {}, {"iters": 10000})),
        (24, agn("ag10k_gated3_rms", gat, {"iters": 10000})),
        (24, agn("ag10k_attn_mem8", {"kind": "attn", "mem": 8}, {"iters": 10000})),
        (24, agn("ag10k_equiv", {"kind": "equiv"}, {"iters": 10000})),
        (24, agn("ag10k_equiv_random", {"kind": "equiv", "heading": "random"}, {"iters": 10000})),
        (24, agn("ag10k_mlph_random", {"kind": "mlph", "heading": "random"}, {"iters": 10000})),
        (24, agn("ag10k_scales2", {"scales": [1.0, 2.0]}, {"iters": 10000})),
    ]
    # H16: how few neighbours can it live with? eps 0.05 is about 7 neighbours, 0.04 about 5 (line of sight limits real bots to 4-6)
    gated_n = {"block": "gated", "depth": 3, "norm": "rms", "zero_head": True}
    for tag, e in (("eps7", 0.05), ("eps5", 0.04)):
        w = {"kind": "wipe", "level": [e, e], "p": 1.0}
        for nm, cell in (("base", {}), ("gated3_rms", gated_n)):
            cfg = variant(f"{tag}_{nm}", cell, {"damage": [w]})
            cfg["eps"] = e
            out.append((29, cfg))
    # does it hold on other shapes? (H9): the plain network, and the all-damage curriculum
    for shape in shapes.SUITE[1:]:
        out.append((45, variant(f"shape_{shape}_base", shape=shape)))
        out.append((46, variant(f"shape_{shape}_all_damage", shape=shape, train={"damage": ALL_DAMAGE})))
    return out


def seen_names():
    names = set()
    for sub in ("master", "dispatched", "pending", "running", "done", "failed"):
        for p in (LAB / "queue" / sub).rglob("*.json"):
            names.add(p.stem.split("_", 1)[1] if p.stem[:2].isdigit() else p.stem)
    runs = LAB.parent / "runs"
    if runs.exists():
        names |= {p.name for p in runs.iterdir() if (p / "result.json").exists()}
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()
    MASTER.mkdir(parents=True, exist_ok=True)
    for sub in ("dispatched",):
        (LAB / "queue" / sub).mkdir(parents=True, exist_ok=True)
    seen, added = seen_names(), 0
    for prio, cfg in experiments():
        cfg["train"]["iters"] = cfg["train"]["iters"] if cfg["train"]["iters"] != 3000 else args.iters
        if cfg["name"] in seen:
            continue
        (MASTER / f"{prio:02d}_{cfg['name']}.json").write_text(json.dumps(cfg, indent=1))
        added += 1
    print(f"queued {added} experiments in {MASTER}")


if __name__ == "__main__":
    main()
