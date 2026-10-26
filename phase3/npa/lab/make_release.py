"""Build phase3/npa/release/: a results table (one row per run), the summary of every run, and the weights of the runs the findings rely on.

    python lab/make_release.py
No GPU needed. Reads runs/*/result.json and params.npz. Never copies the authors' pretrained lizard weights or any CAD mesh.
"""
import csv, json, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NPA = HERE.parent
OUT = NPA / "release"
(OUT / "weights").mkdir(parents=True, exist_ok=True)

NOTES = {   # runs whose numbers need a warning (see lab/FINDINGS.md for each)
    "all_damage_10k": "misnamed: training stopped at about 3,000 of 10,000 iterations (see lab/FINDINGS.md, section on the demo model)",
    "pa8_mem8_all_damage_s1": "diverged (FINDINGS 30)",
    "ag_turn_mlph_random": "old turn rule: any proposal triggered a 28 mm jump, swarm wrecked (FINDINGS 31)",
    "ag_turn_equiv_random": "old turn rule, swarm wrecked (FINDINGS 32)",
    "ag_turn_mlph_scales2": "old turn rule, swarm wrecked (FINDINGS 33)",
    "ag_turn2_mlph_scales2": "never turns (0 turns in 118M bot-ticks), equals a one-line run (FINDINGS 36)",
    "ag_turn2_mlph_random": "never turns, equals a one-line run (FINDINGS 36)",
    "ag_turn2_equiv_random": "turn rate not measured, expected to equal a one-line run",
    "ag_turn3_mlph_scales2": "never turns, equals a one-line run (FINDINGS 42)",
    "ag_turnst2_mlph_scales2": "turns on 1.95% of bot-ticks, swarm degraded (FINDINGS 45)",
    "ag_turnst03_mlph_scales2": "turns on 0.46% of bot-ticks, swarm degraded (FINDINGS 50)",
    "ag_turnst_mlph_scales2": "turn rate not measured, swarm wrecked (FINDINGS 46)",
}
INVALID = ("ag_dualtax_",)
KEEP_WEIGHTS = ["pa8_scales2_all_damage", "pa8_scales2_all_damage_s1", "pa8_scales2_all_damage_s2", "ag_axis_mlph_random", "ag_axis_mlph_scales2",
                "ag_axis_mlph_scales2_s1", "ag_axis_mlph_scales2_20k", "ag_thin_axis_mlph_scales2", "ag_thin_axis_mlph_scales2_s1", "ag_eps7_scales2_warm1",
                "ag_eps7_scales2_warm1_s1", "ag_thin_free_scales2", "ag_thin_free_scales2_s1", "ag_eps7_attn_mem8_warm1", "ag_eps7_attn_mem8_warm1_s1",
                "ag_geo_warm1_equiv_20k_L", "ms4_scales2_pa8_L", "ms4_mem8_pa8_L", "ms4_base_40k_L"]
KINDS = ["clump", "noise", "drag", "cut", "shuffle", "wipe"]

rows, summaries = [], {}
for p in sorted((NPA / "runs").glob("*/result.json")):
    try:
        d = json.loads(p.read_text())
    except ValueError:
        continue
    sm = d.get("summary")
    if not sm:
        continue
    cfg = d.get("config", {}); cell = cfg.get("cell", {}) or {}; tr = cfg.get("train", {}) or {}; emb = cell.get("embody") or {}
    row = {"name": d["name"], "shape": "multi:" + ",".join(cfg["shapes"]) if "shapes" in cfg else cfg.get("shape", d.get("shape", "lizard")),
           "mode": tr.get("mode", "bptt"), "cell_kind": cell.get("kind", "mlp"), "scales": str(cell.get("scales", "")), "heading": cell.get("heading", ""),
           "body_model": "yes" if emb else "no", "one_line_motion": emb.get("axis_lock", ""), "axis_turn": emb.get("axis_turn", ""), "turn_st": emb.get("turn_st", ""),
           "ring_vis": cell.get("ring_vis", ""), "dropout": emb.get("dropout", ""), "particles": cfg.get("particles", ""), "seed": cfg.get("seed", 0),
           "iters_done": d.get("iters_done", ""), "iters_planned": tr.get("iters", ""), "blowups": d.get("blowups", ""),
           "train_minutes": round(d.get("train_minutes") or 0, 1), "params": d.get("params", ""),
           "score": round(sm.get("score", float("nan")), 4), "control_loss": round(sm.get("control_loss", float("nan")), 4),
           "excess": round(sm.get("excess", float("nan")), 4), "finite": sm.get("finite", "")}
    dm = (d.get("eval") or {}).get("damage") or {}
    for k in KINDS:
        kd = dm.get(k)
        if kd and "do_nothing_loss_final" in kd:        # shape-agnostic runs only: gain over doing nothing, small/medium/large
            g = (np.array(kd["do_nothing_loss_final"]) - np.array(kd["loss_final"])).mean(1)
            for lvl, v in zip(("small", "medium", "large"), g):
                row[f"gain_{k}_{lvl}"] = round(float(v), 3)
    note = NOTES.get(d["name"], "")
    if d["name"].startswith(INVALID):
        note = "invalid by construction (forced random turns), ignore (FINDINGS 28)"
    if d.get("iters_done") and tr.get("iters") and d["iters_done"] < 0.5 * tr["iters"]:
        note = (note + "; " if note else "") + "stopped early (blow-ups or timeout)"
    row["note"] = note
    rows.append(row); summaries[d["name"]] = {"config": cfg, "summary": sm, "iters_done": d.get("iters_done"), "blowups": d.get("blowups"), "note": note}

cols = []
for r in rows:
    for k in r:
        if k not in cols:
            cols.append(k)
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
(OUT / "summaries.json").write_text(json.dumps(summaries, separators=(",", ":")))

kept = []
for n in KEEP_WEIGHTS:
    src = NPA / "runs" / n / "params.npz"
    if src.exists():
        shutil.copy(src, OUT / "weights" / f"{n}.npz")
        (OUT / "weights" / f"{n}.config.json").write_text(json.dumps(summaries[n]["config"], indent=1))
        kept.append(n)
print(f"{len(rows)} runs in results.csv, {len(kept)} weight files; skipped weights: {sorted(set(KEEP_WEIGHTS) - set(kept))}")
