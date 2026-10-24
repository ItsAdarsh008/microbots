"""Requeue runs that stopped well short of their iteration budget (blow-up give-up) under the fixed trainer.

    python lab/requeue_broken.py          # writes lab/queue/master/<prio>_<name>_r2.json

A rerun gets "rev": 2 in its config (so the generator does not treat it as a duplicate) and the name suffix _r2.
Runs from the invalid first physical scale (phys_*) are skipped; see hypotheses.md H19.
"""

import copy
import glob
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent
RUNS = LAB.parent / "runs"
MASTER = LAB / "queue" / "master"
MASTER.mkdir(parents=True, exist_ok=True)
done = {Path(p).parent.name for p in glob.glob(str(RUNS / "*/result.json"))}
queued = {p.stem.split("_", 1)[1] for p in (LAB / "queue").rglob("*.json") if p.stem[:2].isdigit()}
n = 0
for p in sorted(glob.glob(str(RUNS / "*/result.json"))):
    d = json.loads(Path(p).read_text())
    cfg, want, got = d.get("config", {}), d.get("config", {}).get("train", {}).get("iters"), d.get("iters_done")
    name = d["name"]
    if not want or got is None or got >= 0.9 * want or name.startswith("phys_") or name.endswith("_r2") or "__" in name:
        continue
    if not d.get("blowups"):
        continue                                   # stopped for another reason (e.g. NaN before the guard existed)
    new = copy.deepcopy(cfg)
    new["name"], new["rev"] = name + "_r2", 2
    new.pop("init_params_b64", None) if "init_params_b64" in new and "__ft_" not in name else None
    if new["name"] in done or new["name"] in queued:
        continue
    (MASTER / f"22_{new['name']}.json").write_text(json.dumps(new, indent=1))
    n += 1
print("requeued", n)
