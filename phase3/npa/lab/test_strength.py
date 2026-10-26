"""Checks for lab/strength.py: triangle rigid, chain floppy, square patch vs braced vs triangular patch. Run: python lab/test_strength.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strength as S

ok = []
def check(name, c, d=""):
    ok.append(bool(c)); print(f"[{'PASS' if c else 'FAIL'}] {name}" + (f"  ({d})" if d else ""))

s = 34.0
tri = np.array([[0, 0], [s, 0], [s / 2, s * np.sqrt(3) / 2]]); tb = S.contact_bonds(tri, 1.2 * s)
chain = np.array([[0, 0], [s, 0], [2 * s, 0], [3 * s, 0]]); cb = S.contact_bonds(chain, 1.2 * s)
check("triangle has 3 bonds", len(tb) == 3)
f, p = S.floppy_modes(tri, tb); check("triangle rigid (0 floppy modes)", f == 0 and p == 1, f"floppy {f}")
f2, _ = S.floppy_modes(chain, cb); check("chain of 4: still 0 floppy modes with rigid (welded) bonds", f2 == 0, f"floppy {f2}")
wt, wc = S.worst_mode(tri, tb), S.worst_mode(chain, cb)
check("triangle stiffer than a chain (worst mode)", wt > wc, f"{wt:.3f} vs {wc:.4f}")
sq = np.array([[i * s, j * s] for i in range(4) for j in range(4)], float)
sqb = S.contact_bonds(sq, 1.1 * s)
hx = np.array([[i * s + (j % 2) * s / 2, j * s * np.sqrt(3) / 2] for i in range(4) for j in range(4)], float)
hxb = S.contact_bonds(hx, 1.1 * s)
check("square patch has degree about 4, hex about 6", abs(S.mean_degree(16, sqb) - 3.0) < 0.6 and S.mean_degree(16, hxb) > 3.8, f"{S.mean_degree(16, sqb):.2f} / {S.mean_degree(16, hxb):.2f}")
q = S.q_rel(sq, sqb, hx, hxb); check("square patch softer than hex reference (Q_rel < 1)", q < 1.0, f"Q_rel {q:.3f}")
split = np.vstack([tri, tri + 500]); sb = S.contact_bonds(split, 1.2 * s)
check("two far-apart triangles: Q = 0 (split)", S.worst_mode(split, sb) < 1e-9)
check("rigid_fraction of a triangle is 1, of two split triangles 0.5 or 0", S.rigid_fraction(tri, tb) == 1.0 and S.rigid_fraction(split, sb) in (0.0, 0.5))
check("a chain has all bonds as bridges", S.critical_fraction(4, cb) == 1.0)
check("a triangle has no bridges", S.critical_fraction(3, tb) == 0.0)
print(f"{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
