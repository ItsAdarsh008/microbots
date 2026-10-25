"""Checks for the linked lattice swarm. Run from the repo root:

    python linking/checks.py

Each check prints PASS or FAIL. The last line says how many passed.
"""

import numpy as np

import lattice as L

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# 1. The move rules on a tiny hand-made example: one bot on top of another.
occ = np.zeros(L.GRID, dtype=bool)
occ[5, 5, 0] = occ[5, 5, 1] = True
moves = {(k, c) for k, c, _ in L.legal_moves(occ, (5, 5, 1))}
pivots = {c for k, c in moves if k == "pivot"}
check("a bot on top of one other bot can only pivot down its sides",
      pivots == {(6, 5, 0), (4, 5, 0), (5, 6, 0), (5, 4, 0)} and
      all(k == "pivot" for k, _ in moves),
      ", ".join(str(tuple(int(v) for v in c)) for c in sorted(pivots)))

moves = {(k, c) for k, c, _ in L.legal_moves(occ, (5, 5, 0))}
check("a bot on the floor can slide along the floor",
      ("slide", (6, 5, 0)) in moves)

occ[6, 5, 1] = True        # block the corner the top bot would swing through
blocked = {c for k, c, _ in L.legal_moves(occ, (5, 5, 1)) if k == "pivot"}
check("a pivot is not allowed when the corner it swings through is full",
      (6, 5, 0) not in blocked)

# 2. Full builds: every shape, a few random seeds.
for shape in L.SHAPES:
    for seed in range(3):
        target, frames, swarm = L.run(shape, seed=seed)
        ok = frames[-1]["filled"] == 1.0
        connected = all(f["connected"] for f in frames)
        check(f"{shape} (seed {seed}) is fully built and stays in one piece",
              ok and connected, f"{len(frames)} steps, {swarm.moves_made} moves")

# 3. Every move is a real slide or pivot, bots never share a cell, and
#    locked bots never move.
target, frames, swarm = L.run("arch", seed=1)
bad_moves = overlaps = locked_moved = 0
for prev, f in zip(frames, frames[1:]):
    step = f["pos"] - f["from"]
    for b in np.where(f["kind"] == 1)[0]:
        bad_moves += np.abs(step[b]).sum() != 1
    for b in np.where(f["kind"] == 2)[0]:
        d1 = f["via"][b] - f["from"][b]          # first half of the roll
        d2 = f["pos"][b] - f["via"][b]           # second half
        bad_moves += not (np.abs(d1).sum() == 1 and np.abs(d2).sum() == 1
                          and d1 @ d2 == 0)
        bad_moves += bool(prev["occ"][tuple(f["via"][b])])  # swung through a full cell
    for b in np.where(f["kind"] > 0)[0]:
        bad_moves += bool(prev["occ"][tuple(f["pos"][b])])  # landed where a bot just was
    cells = {tuple(p) for p in f["pos"][f["alive"]]}
    overlaps += len(cells) != f["alive"].sum()
    was_locked = prev["bot_locked"]
    locked_moved += int((np.abs(f["pos"] - prev["pos"]).sum(axis=1)[was_locked] > 0).sum())
check("every move is one slide or one pivot, into cells that were empty", bad_moves == 0,
      f"{bad_moves} bad")
check("no two bots ever share a cell", overlaps == 0)
check("locked bots never move", locked_moved == 0)

# 4. Damage and repair: the shape is rebuilt from spare bots.
for shape, (make_region, extra) in {
        "sheet": (lambda: L.cut_top("sheet", 5), 40),
        "arch": (lambda: L.cut_box(14, 18, 0, 16, 0, 14), 30)}.items():
    target, frames, swarm = L.run(shape, damage_region=make_region(), extra=extra)
    hit = [i for i, f in enumerate(frames) if f["damaged"]]
    ok = len(hit) == 1 and frames[-1]["filled"] == 1.0
    check(f"{shape}: after losing {frames[-1]['lost']} bots, the shape is rebuilt",
          ok and all(f["connected"] for f in frames),
          f"cut at step {hit[0] if hit else '?'}, done at step {len(frames) - 1}")

print(f"\n{sum(results)}/{len(results)} checks passed")
