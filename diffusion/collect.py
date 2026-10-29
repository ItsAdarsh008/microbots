"""Pull finished runs from the GPU machines into results/ and print one table.
    python collect.py
"""
import glob, json, os, subprocess
HOSTS = ["ubuntu@129.213.131.224", "ubuntu@129.146.37.80"]
HERE = os.path.dirname(os.path.abspath(__file__))
for h in HOSTS:
    subprocess.run(["rsync", "-az", "--include=*/", "--include=result.json", "--include=samples.png", "--include=traj.gif",
                    "--include=log.txt", "--exclude=*", f"{h}:~/diffusion/runs/", f"{HERE}/results/"], check=False)
rows = []
for f in sorted(glob.glob(f"{HERE}/results/*/result.json")):
    r = json.load(open(f)); c = r["config"]; K = f"K{max(c['eval_steps'])}"
    row = dict(name=c["name"], loss=r["final_loss"], mins=r["train_seconds"] / 60)
    for s in ("val", "ood_family", "emoji"):
        t = r["sets"][s]
        row[s] = (t[K]["w2"], t[K]["skill"], t[K]["cover"], t[K]["coll"])
        row[s + "_cl"] = t["classical"]["w2"] if "classical" in t else float("nan")
    row["conn"] = r["stats"]["connected_frac"]["gen"]; row["holes_w1"] = r["stats"]["holes"]["w1"]
    row["iou"] = r["stats"]["best_iou_train"]["gen"]; row["iou_real"] = r["stats"]["best_iou_train"]["real"]
    row["wrong"] = r.get("control_wrong_map")
    rows.append(row)
print(f"{'run':34s} {'val w2/skill/cov/coll':28s} {'ood-family':28s} {'emoji':28s} conn  iou(real)")
for r in rows:
    f = lambda t: f"{t[0]:5.2f} {t[1]:4.2f} {t[2]:4.2f} {t[3]:4.2f}"
    print(f"{r['name']:34s} {f(r['val']):28s} {f(r['ood_family']):28s} {f(r['emoji']):28s} {r['conn']:.2f}  {r['iou']:.2f}({r['iou_real']:.2f})")
if rows:
    print("classical baseline w2 (val / ood / emoji):", " / ".join(f"{rows[0][k]:.2f}" for k in ("val_cl", "ood_family_cl", "emoji_cl")))
    for r in rows:
        if r["wrong"]:
            print(f"  wrong-map control {r['name']}: w2 vs fed map {r['wrong']['w2_vs_fed_map']:.2f}, vs the other shape {r['wrong']['w2_vs_other_shape']:.2f}")
