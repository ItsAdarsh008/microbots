"""Parametric CAD model of one microbot (v1), in CadQuery.

One bot = a spherical body with two arms on its poles. Each arm has three joints:
  azimuth (turret spinning about the body's pole axis),
  pitch   (clevis hinge on the turret, pin in double shear),
  roll    (the tip collar spinning about the arm's own axis).
Every tip and the four equatorial body ports carry the same genderless
connector: a Hirth face (radial V-teeth) around an electro-permanent-magnet
(EPM) pole. Any connector mates with any other one, at any roll that is a
multiple of 360/N degrees.

Units: millimetres, degrees. Every dimension lives in `Params`.

Frames (body frame):
  +Z / -Z  poles, where the two arms sit ("top" arm on +Z, "bottom" on -Z)
  +-X, +-Y equatorial docking ports
  A connector frame has its origin at the centre of the teeth pitch plane,
  +z pointing out of the face and +x the reference ("flip") axis. Two
  connectors mate when frame_B = frame_A * Rx(180) * Rz(k * 360/N).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np
from OCP.gp import gp_Trsf


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    # body
    body_r: float = 15.0          # body sphere radius (Ø30)
    wall: float = 1.2             # shell wall
    pole_flat: float = 13.0       # polar flat height (turret seat)
    spigot_r: float = 2.5         # turret spigot (holds the coaxial drive shafts)
    shaft_bore_r: float = 1.3
    # connector (shared by tips and ports)
    conn_r: float = 5.0           # outer radius of the toothed face (Ø10)
    conn_r_in: float = 2.5        # inner radius of the teeth = EPM pole radius
    teeth: int = 24               # Hirth teeth -> 15 deg roll indexing
    tooth_h: float = 0.8          # tooth height (crest to root)
    backlash_deg: float = 0.35    # angular clearance per flank
    pole_recess: float = 0.1      # EPM pole face sits this far below the pitch plane
    epm_r: float = 2.5            # EPM pocket radius
    epm_depth: float = 3.0
    port_d: float = 14.0          # body centre -> port pitch plane
    # turret + clevis
    turret_r: float = 7.0
    turret_hw: float = 3.3        # turret is a bar: half-width across the swing plane
    turret_t: float = 2.0
    hub_r: float = 3.0            # arm hub radius around the pitch pin
    hub_w: float = 6.0            # hub width along the pin
    ear_t: float = 1.6            # clevis ear thickness
    ear_gap: float = 0.2          # side clearance hub <-> ear
    pin_r: float = 1.0            # Ø2 steel dowel
    hub_clear: float = 0.5        # hub <-> turret top clearance
    # arm
    arm_len: float = 22.0         # pitch pin -> tip pitch plane
    neck_t: float = 4.0           # neck thickness (in the pitch plane)
    tube_r: float = 4.0           # motor housing tube radius (Ø8)
    tube_bore_r: float = 3.0      # Ø6 bore for a Ø4-6 mm roll gearmotor
    collar_len: float = 4.0       # roll collar length behind the face
    collar_gap: float = 0.2

    @property
    def pin_z(self) -> float:
        return self.pole_flat + self.turret_t + self.hub_clear + self.hub_r

    @property
    def span(self) -> float:
        """Tip-to-tip distance with both arms straight."""
        return 2 * (self.pin_z + self.arm_len)

    @property
    def pitch_deg(self) -> float:
        return 360.0 / self.teeth


# ----------------------------------------------------------------------------
# Rigid transforms (4x4 numpy) <-> CadQuery
# ----------------------------------------------------------------------------
def T(x=0.0, y=0.0, z=0.0):
    M = np.eye(4)
    M[:3, 3] = [x, y, z]
    return M


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    M[i, i], M[i, j], M[j, i], M[j, j] = c, -s, s, c
    return M


def Rx(d):
    return _rot("x", d)


def Ry(d):
    return _rot("y", d)


def Rz(d):
    return _rot("z", d)


FLIP = Rx(180.0)


def to_loc(M) -> cq.Location:
    t = gp_Trsf()
    t.SetValues(*M[0, :4], *M[1, :4], *M[2, :4])
    return cq.Location(t)


def place(shape: cq.Shape, M) -> cq.Shape:
    return shape.moved(to_loc(M))


# ----------------------------------------------------------------------------
# Genderless Hirth connector
# ----------------------------------------------------------------------------
def hirth_teeth(P: Params) -> cq.Shape:
    """Radial V-teeth in the annulus conn_r_in..conn_r, z in [-h/2, h/2].

    Crests sit at (k + 1/4) * pitch so that a copy flipped about +x lands its
    crests exactly in the roots: zero relative roll is a valid mate, and so is
    every multiple of the pitch (15 deg for 24 teeth).
    """
    p = 2 * math.pi / P.teeth
    w = p / 2 - math.radians(P.backlash_deg)        # half base width (angle)
    h2 = P.tooth_h / 2
    r0, r1 = P.conn_r_in * 0.98, P.conn_r * 1.06    # overshoot, trimmed below
    teeth = []
    for k in range(P.teeth):
        th = (k + 0.25) * p
        wires = []
        for r in (r0, r1):
            pts = [
                cq.Vector(r * math.cos(th - w), r * math.sin(th - w), -h2),
                cq.Vector(r * math.cos(th), r * math.sin(th), h2),
                cq.Vector(r * math.cos(th + w), r * math.sin(th + w), -h2),
            ]
            wires.append(cq.Wire.makePolygon(pts, close=True))
        teeth.append(cq.Solid.makeLoft(wires, ruled=True))
    ring = cq.Workplane().add(teeth).combine().val()
    trim = (
        cq.Workplane().workplane(offset=-h2 - 0.01)
        .circle(P.conn_r).circle(P.conn_r_in).extrude(P.tooth_h + 0.02).val()
    )
    return ring.intersect(trim)


def connector(P: Params, depth: float) -> cq.Shape:
    """Connector in its own frame: teeth + a solid backing of `depth` below the
    tooth roots, with the EPM pocket. The EPM pole fills the centre up to
    pole_recess below the pitch plane (modelled as part of the face)."""
    h2 = P.tooth_h / 2
    back = (
        cq.Workplane().workplane(offset=-h2 - depth)
        .circle(P.conn_r).extrude(depth).val()
    )
    pole = (
        cq.Workplane().workplane(offset=-h2)
        .circle(P.conn_r_in).extrude(h2 - P.pole_recess).val()
    )
    return back.fuse(pole).fuse(hirth_teeth(P)).clean()


# ----------------------------------------------------------------------------
# Parts (each built in the body frame for the TOP arm, all joints at zero)
# ----------------------------------------------------------------------------
PORT_DIRS = {"+X": 0.0, "+Y": 90.0, "-X": 180.0, "-Y": 270.0}


def port_frame(P: Params, name: str):
    """Port connector frame: local z = outward normal, local x = body +Z."""
    R0 = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    return Rz(PORT_DIRS[name]) @ T(P.port_d, 0, 0) @ R0


def body(P: Params) -> cq.Shape:
    h2 = P.tooth_h / 2
    shell = cq.Workplane().sphere(P.body_r).val().cut(
        cq.Workplane().sphere(P.body_r - P.wall).val())
    # polar flats (turret seats) with a solid seat plate
    big = 4 * P.body_r
    for s in (1, -1):
        cutter = cq.Workplane().box(big, big, big).translate((0, 0, s * (P.pole_flat + big / 2))).val()
        shell = shell.cut(cutter)
        seat_z0 = P.pole_flat - 2.0 if s > 0 else -P.pole_flat
        seat = cq.Workplane().workplane(offset=seat_z0).circle(
            math.sqrt(P.body_r ** 2 - P.pole_flat ** 2)).extrude(2.0).val()
        shell = shell.fuse(seat.intersect(cq.Workplane().sphere(P.body_r).val()))
        spig_z0 = P.pole_flat if s > 0 else -P.pole_flat - P.turret_t
        spigot = cq.Workplane().workplane(offset=spig_z0).circle(P.spigot_r).extrude(P.turret_t).val()
        shell = shell.fuse(spigot)
    shaft = cq.Workplane().workplane(offset=-P.body_r - 5).circle(P.shaft_bore_r).extrude(2 * P.body_r + 10).val()
    shell = shell.cut(shaft)
    # equatorial ports: flat-cut the sphere at the tooth roots, add connectors
    depth = P.epm_depth + 0.6
    for name in PORT_DIRS:
        F = port_frame(P, name)
        flat_cut = cq.Workplane().workplane(offset=-h2).rect(big, big).extrude(big).val()
        shell = shell.cut(place(flat_cut, F))
        boss = connector(P, depth)
        shell = shell.fuse(place(boss, F))
        pocket = cq.Workplane().workplane(offset=-h2 - depth - 0.01).circle(P.epm_r - 0.6).extrude(depth - 0.6).val()
        shell = shell.cut(place(pocket, F))
    return shell.clean()


def turret(P: Params) -> cq.Shape:
    z0 = P.pole_flat
    # a bar, not a full disc, so the arm can swing down past the turret
    disk = (cq.Workplane().workplane(offset=z0).circle(P.turret_r)
            .circle(P.spigot_r + 0.1).extrude(P.turret_t).val()
            .intersect(cq.Workplane().box(2 * P.turret_r, 2 * P.turret_hw, 50).val()))
    x_in = P.hub_w / 2 + P.ear_gap
    ear_h = P.pin_z - (z0 + P.turret_t)
    ears = None
    for s in (1, -1):
        x0 = x_in if s > 0 else -x_in - P.ear_t
        prof = (cq.Workplane("YZ").workplane(offset=x0)
                .center(0, z0 + P.turret_t)
                .rect(2 * P.hub_r, ear_h, centered=(True, False)).extrude(P.ear_t).val())
        cap = (cq.Workplane("YZ").workplane(offset=x0)
               .center(0, P.pin_z).circle(P.hub_r).extrude(P.ear_t).val())
        e = prof.fuse(cap)
        ears = e if ears is None else ears.fuse(e)
    t = disk.fuse(ears)
    pin_hole = (cq.Workplane("YZ").workplane(offset=-10).center(0, P.pin_z)
                .circle(P.pin_r + 0.02).extrude(20).val())
    return t.cut(pin_hole).clean()


def pin(P: Params) -> cq.Shape:
    L = P.hub_w + 2 * (P.ear_gap + P.ear_t)
    return (cq.Workplane("YZ").workplane(offset=-L / 2).center(0, P.pin_z)
            .circle(P.pin_r).extrude(L).val())


def arm(P: Params) -> cq.Shape:
    zp = P.pin_z
    hub = (cq.Workplane("YZ").workplane(offset=-P.hub_w / 2).center(0, zp)
           .circle(P.hub_r).extrude(P.hub_w).val())
    neck_top = zp + P.hub_r + 2.0
    neck = (cq.Workplane().workplane(offset=zp)
            .rect(P.hub_w, P.neck_t).extrude(neck_top - zp).val())
    tube_z1 = zp + P.arm_len - P.tooth_h / 2 - P.collar_len - P.collar_gap
    tube = (cq.Workplane().workplane(offset=neck_top - 0.5)
            .circle(P.tube_r).extrude(tube_z1 - neck_top + 0.5).val())
    bore = (cq.Workplane().workplane(offset=neck_top + 0.8)
            .circle(P.tube_bore_r).extrude(tube_z1 - neck_top).val())
    a = hub.fuse(neck).fuse(tube).cut(bore)
    pin_hole = (cq.Workplane("YZ").workplane(offset=-10).center(0, zp)
                .circle(P.pin_r + 0.02).extrude(20).val())
    return a.cut(pin_hole).clean()


def tip(P: Params) -> cq.Shape:
    """Roll collar + connector, in the tip frame (pitch plane at z=0)."""
    t = connector(P, P.collar_len)
    pocket = (cq.Workplane().workplane(offset=-P.tooth_h / 2 - P.collar_len - 0.01)
              .circle(P.epm_r).extrude(P.epm_depth).val())
    return t.cut(pocket).clean()


def tip_local(P: Params):
    """Tip frame relative to the arm frame (top arm, zero pitch)."""
    return T(0, 0, P.pin_z + P.arm_len)


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
@dataclass
class Pose:
    """Joint angles of one arm (deg). Pitch tilts the arm from its pole toward
    the turret's +Y; azimuth spins the turret; roll spins the tip collar."""
    az: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class BotPose:
    top: Pose = field(default_factory=Pose)
    bottom: Pose = field(default_factory=Pose)


ARMS = {"top": np.eye(4), "bottom": Rx(180.0)}


def arm_frames(P: Params, which: str, pose: Pose):
    """Return (turret_M, arm_M, tip_M) in the body frame."""
    base = ARMS[which]
    Mt = base @ Rz(pose.az)
    Ma = Mt @ T(0, 0, P.pin_z) @ Rx(-pose.pitch) @ T(0, 0, -P.pin_z)
    Mtip = Ma @ tip_local(P) @ Rz(pose.roll)
    return Mt, Ma, Mtip


def conn_frame(P: Params, bp: BotPose, name: str):
    """Connector frame in the body frame: 'top', 'bottom' or a port name."""
    if name in ARMS:
        return arm_frames(P, name, getattr(bp, name))[2]
    return port_frame(P, name)


def mate(P: Params, M_world_A, bpB: BotPose, connB: str, k: int = 0):
    """World transform of bot B's body so that its connector connB mates the
    world connector frame M_world_A, with k teeth of relative roll."""
    MB = conn_frame(P, bpB, connB)
    return M_world_A @ FLIP @ Rz(k * P.pitch_deg) @ np.linalg.inv(MB)


# ----------------------------------------------------------------------------
# Building one posed bot
# ----------------------------------------------------------------------------
class PartCache:
    def __init__(self, P: Params):
        self.P = P
        self._c = {}

    def __getitem__(self, name):
        if name not in self._c:
            self._c[name] = {"body": body, "turret": turret, "arm": arm,
                             "tip": tip, "pin": pin}[name](self.P)
        return self._c[name]


def bot_parts(parts: PartCache, bp: BotPose, M_world=np.eye(4)):
    """List of (name, shape) for one bot placed at M_world."""
    P = parts.P
    out = [("body", place(parts["body"], M_world))]
    for which in ARMS:
        Mt, Ma, Mtip = arm_frames(P, which, getattr(bp, which))
        out.append((f"turret_{which}", place(parts["turret"], M_world @ Mt)))
        out.append((f"pin_{which}", place(parts["pin"], M_world @ Ma)))
        out.append((f"arm_{which}", place(parts["arm"], M_world @ Ma)))
        out.append((f"tip_{which}", place(parts["tip"], M_world @ Mtip)))
    return out
