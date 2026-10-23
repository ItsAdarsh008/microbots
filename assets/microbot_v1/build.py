"""Build every export, check the geometry, and write report.json.

    python build.py            # STEP + STL + renders + report (a few minutes)
    python build.py --fast     # skip renders

Outputs land next to this file in step/, stl/, renders/ and report.json.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import cadquery as cq
import numpy as np

from microbot import (ARMS, FLIP, BotPose, Params, PartCache, Pose, Rz, T,
                      arm_frames, bot_parts, conn_frame, mate, place)

HERE = Path(__file__).resolve().parent
STEP, STL, IMG = HERE / "step", HERE / "stl", HERE / "renders"
for d in (STEP, STL, IMG):
    d.mkdir(exist_ok=True)

P = Params()
parts = PartCache(P)

# ----------------------------------------------------------------------------
# Mass model: printed parts from CAD volume, bought parts as point masses
# ----------------------------------------------------------------------------
RHO = {"body": 1.01, "turret": 1.01, "arm": 1.01, "tip": 1.01, "pin": 7.85}  # g/cm3 (PA12, steel)
MATERIAL = {"body": "PA12 (MJF) or tough SLA resin", "turret": "PA12", "arm": "PA12",
            "tip": "PA12 / SLA (fine teeth)", "pin": "hardened steel dowel"}
# (name, grams, position in body frame or arm frame)
BODY_COMPONENTS = [
    ("LiPo 3.7 V ~40 mAh", 1.4, (0, 0, 0)),
    ("PCB: MCU + IMU + motor drivers + IR/BLE", 1.2, (0, 0, -3.5)),
    ("gearmotor Ø4 (differential, top A)", 1.1, (0, 3, 7)),
    ("gearmotor Ø4 (differential, top B)", 1.1, (0, -3, 7)),
    ("gearmotor Ø4 (differential, bottom A)", 1.1, (0, 3, -7)),
    ("gearmotor Ø4 (differential, bottom B)", 1.1, (0, -3, -7)),
    ("EPM port +X", 0.5, (12, 0, 0)), ("EPM port -X", 0.5, (-12, 0, 0)),
    ("EPM port +Y", 0.5, (0, 12, 0)), ("EPM port -Y", 0.5, (0, -12, 0)),
    ("bevel differential gears x2", 0.6, (0, 0, 0)),
]
ARM_COMPONENTS = [  # in the arm frame (top arm, zero pitch)
    ("roll gearmotor Ø4 (in arm tube)", 1.0, (0, 0, P.pin_z + 11)),
    ("EPM in tip", 0.5, (0, 0, P.pin_z + P.arm_len - 2.5)),
]


def mass_props(bp: BotPose):
    total, moment = 0.0, np.zeros(3)
    rows = []
    for name, shp in bot_parts(parts, bp):
        key = name.split("_")[0]
        m = shp.Volume() / 1000 * RHO[key]
        c = np.array(cq.Shape.centerOfMass(shp).toTuple())
        total += m
        moment += m * c
        rows.append((name, m))
    for name, m, pos in BODY_COMPONENTS:
        total += m
        moment += m * np.array(pos)
        rows.append((name, m))
    for which in ARMS:
        Ma = arm_frames(P, which, getattr(bp, which))[1]
        for name, m, pos in ARM_COMPONENTS:
            total += m
            moment += m * (Ma @ np.array([*pos, 1.0]))[:3]
            rows.append((f"{name} [{which}]", m))
    return total, moment / total, rows


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def overlap(a: cq.Shape, b: cq.Shape) -> float:
    ba, bb = a.BoundingBox(), b.BoundingBox()
    if (ba.xmax < bb.xmin or bb.xmax < ba.xmin or ba.ymax < bb.ymin or bb.ymax < ba.ymin
            or ba.zmax < bb.zmin or bb.zmax < ba.zmin):
        return 0.0
    return a.intersect(b).Volume()


def interference(named_shapes, tol=1e-3):
    hits = []
    for (na, a), (nb, b) in itertools.combinations(named_shapes, 2):
        v = overlap(a, b)
        if v > tol:
            hits.append((na, nb, round(v, 4)))
    return hits


def pitch_limit(step=1.0):
    """Largest pitch (deg) at which the arm + tip clear the body and turret."""
    fixed = parts["body"].fuse(parts["turret"])
    for th in np.arange(0, 181, step):
        _, Ma, Mtip = arm_frames(P, "top", Pose(pitch=float(th)))
        moving = [place(parts["arm"], Ma), place(parts["tip"], Mtip)]
        if any(overlap(fixed, m) > 1e-3 for m in moving):
            return float(th - step)
    return 180.0


def mate_residual(MA, MB):
    """How far two world connector frames are from a valid mate:
    (position error mm, axis error deg, roll error deg to nearest tooth)."""
    R = np.linalg.inv(MA @ FLIP) @ MB      # should be Rz(k * pitch)
    pos = float(np.linalg.norm(R[:3, 3]))
    axis = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll_err = abs((roll + P.pitch_deg / 2) % P.pitch_deg - P.pitch_deg / 2)
    return pos, axis, roll_err


# ----------------------------------------------------------------------------
# Poses and demo structures
# ----------------------------------------------------------------------------
def v_pose(inter_arm_deg: float) -> BotPose:
    """Both arms bent toward world +Y so the angle between them is inter_arm_deg."""
    t = (180.0 - inter_arm_deg) / 2
    return BotPose(top=Pose(pitch=t), bottom=Pose(az=180.0, pitch=t))


def chain(n: int, inter_arm_deg: float, k: int):
    """n bots, each bottom tip docked to the next one's top tip with k teeth of
    relative roll. k = 12 (180 deg) makes every body turn the same way (a ring);
    k = 0 makes them alternate (a zigzag)."""
    bp = v_pose(inter_arm_deg)
    Ms = [np.eye(4)]
    for _ in range(n - 1):
        Ms.append(mate(P, Ms[-1] @ conn_frame(P, bp, "bottom"), bp, "top", k))
    return bp, Ms


def ring(n: int):
    """n bots tip-to-tip in a closed planar polygon (bodies at the corners)."""
    bp, Ms = chain(n, 180.0 - 360.0 / n, P.teeth // 2)
    closure = mate_residual(Ms[-1] @ conn_frame(P, bp, "bottom"), Ms[0] @ conn_frame(P, bp, "top"))
    return [(bp, M) for M in Ms], {"closure_mm_deg_deg": [round(x, 6) for x in closure]}


def column_block(nx=2, ny=2, nz=2):
    """Bodies docked port-to-port in each layer, straight arms docked
    tip-to-tip between layers: every body uses all 6 connectors."""
    bp = BotPose()
    Ms = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (i, j, k) == (0, 0, 0):
                    M = np.eye(4)
                elif i > 0:
                    M = mate(P, Ms[(i - 1, j, k)] @ conn_frame(P, bp, "+X"), bp, "-X")
                elif j > 0:
                    M = mate(P, Ms[(i, j - 1, k)] @ conn_frame(P, bp, "+Y"), bp, "-Y")
                else:
                    M = mate(P, Ms[(i, j, k - 1)] @ conn_frame(P, bp, "top"), bp, "bottom")
                Ms[(i, j, k)] = M
    # verify every intended contact, including the ones not used to place bots
    res = []
    for (i, j, k), M in Ms.items():
        for d, (a, b) in {(1, 0, 0): ("+X", "-X"), (0, 1, 0): ("+Y", "-Y"), (0, 0, 1): ("top", "bottom")}.items():
            nb = (i + d[0], j + d[1], k + d[2])
            if nb in Ms:
                res.append(mate_residual(M @ conn_frame(P, bp, a), Ms[nb] @ conn_frame(P, bp, b)))
    worst = [round(max(r[q] for r in res), 6) for q in range(3)]
    return [(bp, M) for M in Ms.values()], {"contacts": len(res), "worst_mm_deg_deg": worst}


def zigzag(n=5, inter_arm_deg=120.0):
    bp, Ms = chain(n, inter_arm_deg, 0)
    return [(bp, M) for M in Ms], {}


def t_junction():
    """Bot B's top tip docked into bot A's +X body port (branching node)."""
    A = BotPose()
    B = BotPose(top=Pose(), bottom=Pose(az=0, pitch=40))
    MB = mate(P, conn_frame(P, A, "+X"), B, "top")
    return [(A, np.eye(4)), (B, MB)], {}


def assembly_shapes(bots):
    named = []
    for b, (bp, M) in enumerate(bots):
        named += [(f"bot{b}.{n}", s) for n, s in bot_parts(parts, bp, M)]
    return named


COLORS = {"body": (0.12, 0.12, 0.14), "turret": (0.75, 0.42, 0.10), "arm": (0.20, 0.22, 0.26),
          "tip": (0.80, 0.80, 0.82), "pin": (0.6, 0.6, 0.6)}


def export_assembly(name, named):
    assy = cq.Assembly(name=name)
    for n, s in named:
        key = n.split(".")[-1].split("_")[0]
        assy.add(s, name=n.replace(".", "_"), color=cq.Color(*COLORS[key]))
    assy.export(str(STEP / f"{name}.step"))
    cq.exporters.export(cq.Compound.makeCompound([s for _, s in named]), str(STL / f"{name}.stl"),
                        tolerance=0.05, angularTolerance=0.2)


# ----------------------------------------------------------------------------
# Rendering (matplotlib, no display needed)
# ----------------------------------------------------------------------------
def render(named, path, title, elev=22, azim=-60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
    tris, cols = [], []
    for n, s in named:
        key = n.split(".")[-1].split("_")[0]
        vs, fs = s.tessellate(0.15, 0.4)
        V = np.array([v.toTuple() for v in vs])
        for f in fs:
            t = V[list(f)]
            nrm = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            sh = 0.35 + 0.65 * abs(nrm @ light) / ln
            tris.append(t)
            cols.append(np.clip(np.array(COLORS[key]) * sh + 0.08 * sh, 0, 1))
    allp = np.concatenate(tris)
    lo, hi = allp.min(0), allp.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    fig = plt.figure(figsize=(7, 7), dpi=130)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolor="none"))
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


# ----------------------------------------------------------------------------
def main(fast=False):
    report = {"params": {k: getattr(P, k) for k in P.__dataclass_fields__},
              "derived": {"pin_z_mm": P.pin_z, "span_straight_mm": P.span,
                          "roll_index_deg": P.pitch_deg}}

    # single parts
    for n in ("body", "turret", "arm", "tip", "pin"):
        cq.exporters.export(parts[n], str(STEP / f"part_{n}.step"))
        cq.exporters.export(parts[n], str(STL / f"part_{n}.stl"), tolerance=0.02, angularTolerance=0.1)

    # connector mate check (tip against flipped tip at several rolls)
    tipS = parts["tip"]
    report["connector_overlap_mm3"] = {
        f"{r:g}deg": round(overlap(tipS, place(tipS, FLIP @ Rz(r))), 4)
        for r in (0, 7.5, 15, 45, 90)}

    # range of motion
    pmax = pitch_limit()
    report["range_of_motion"] = {
        "azimuth_deg": "continuous mechanically; +-180 with a cable service loop",
        "pitch_deg": [-pmax, pmax], "roll_deg": "continuous",
        "tip_direction_coverage_per_arm": f"cone of half-angle {pmax:.0f} deg about its pole "
                                          f"= {100 * (1 - math.cos(math.radians(pmax))) / 2:.0f}% of all directions"}

    poses = {"straight": BotPose(), "V120": v_pose(120), "V60": v_pose(60),
             "U_grip": v_pose(180 - 2 * min(100, pmax)),
             "twisted": BotPose(top=Pose(az=40, pitch=70, roll=30), bottom=Pose(az=-110, pitch=55))}
    report["poses"], report["interference"] = {}, {}
    for name, bp in poses.items():
        m, com, rows = mass_props(bp)
        report["poses"][name] = {"mass_g": round(m, 2), "com_mm": [round(x, 2) for x in com],
                                 "com_offset_from_body_centre_mm": round(float(np.linalg.norm(com)), 2)}
        named = [(f"bot0.{n}", s) for n, s in bot_parts(parts, bp)]
        report["interference"][f"bot_{name}"] = interference(named)
        export_assembly(f"bot_{name}", named)
        if not fast:
            render(named, IMG / f"bot_{name}.png", f"microbot v1 · {name}")
    report["mass_breakdown_g"] = {n: round(m, 2) for n, m in mass_props(BotPose())[2]}

    demos = {"ring3_triangle": ring(3), "ring4_square": ring(4), "ring6_hexagon": ring(6),
             "zigzag5": zigzag(),
             "block_2x2x2": column_block(), "t_junction": t_junction()}
    report["structures"] = {}
    for name, (bots, info) in demos.items():
        named = assembly_shapes(bots)
        hits = [h for h in interference(named)]
        report["structures"][name] = {"bots": len(bots), **info, "interference": hits}
        export_assembly(name, named)
        if not fast:
            render(named, IMG / f"{name}.png", f"{name} · {len(bots)} bots",
                   elev=0 if name.startswith(("ring", "zig")) else 22, azim=0 if name.startswith(("ring", "zig")) else -60)

    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
