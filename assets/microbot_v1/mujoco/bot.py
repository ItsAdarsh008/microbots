"""Articulated MuJoCo model of one microbot v1, built from the CAD meshes (../stl/part_*.stl).

    from bot import BotSim
    sim = BotSim(friction=0.6)
    sim.goal(pitch_top=43); sim.run(1.0)
    print(sim.body_xy(), sim.tip_xy("top"))

The joint chain is the CAD's (microbot.py, arm_frames): the turret spins about the body +Z (azimuth), the arm hinges
about the turret's +X axis through a pin 18.5 mm along the pole (pitch, MuJoCo angle = -pitch because the CAD uses
Rx(-pitch)), and the tip collar rolls about the arm axis. The bottom arm is the top one turned 180 degrees about X.
Units: SI inside MuJoCo (meshes scaled mm to m); the helpers speak mm and degrees.

What is modelled: the real mesh geometry for contact (each part's convex hull), masses summing to 19.3 g, torque-limited
position servos (15 mN*m, the design note's 10 to 20 mN*m class for a 4 mm gearmotor), joint targets slewed at 120 deg/s,
Coulomb friction on a flat floor.
What is not: the toothed face (the hull is a flat disc), cable limits, motor dynamics and backlash, floor compliance,
the electro-permanent magnets. Treat results as an upper bound on how well the gait works.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
STL = HERE.parent / "stl"
PIN_Z, ARM_LEN, BODY_R = 18.5, 22.0, 15.0         # mm, from report.json
JOINT_SPEED = 120.0                                # deg/s
SERVO_TORQUE = 0.015                               # N*m
MASS = {"body": 0.0169, "turret": 0.0003, "arm": 0.0006, "tip": 0.0003}     # kg; total 19.3 g

JOINTS = ("az_top", "pitch_top", "roll_top", "az_bot", "pitch_bot", "roll_bot")


def _xml(friction):
    mm = "0.001 0.001 0.001"
    def arm(side, quat):
        return f"""
      <body name="turret_{side}" quat="{quat}">
        <joint name="az_{side}" type="hinge" axis="0 0 1" damping="0.002" armature="2e-6"/>
        <geom type="mesh" mesh="turret" mass="{MASS['turret']}" rgba="0.55 0.55 0.6 1" friction="{friction} 0.005 0.0001"/>
        <body name="arm_{side}" pos="0 0 {PIN_Z/1000}">
          <joint name="pitch_{side}" type="hinge" axis="1 0 0" range="-100 100" limited="true" damping="0.002" armature="2e-6"/>
          <geom type="mesh" mesh="arm" pos="0 0 {-PIN_Z/1000}" mass="{MASS['arm']}" rgba="0.8 0.8 0.85 1" friction="{friction} 0.005 0.0001"/>
          <body name="tip_{side}" pos="0 0 {ARM_LEN/1000}">
            <joint name="roll_{side}" type="hinge" axis="0 0 1" damping="0.001" armature="1e-6"/>
            <geom type="mesh" mesh="tip" mass="{MASS['tip']}" rgba="0.9 0.5 0.2 1" friction="{friction} 0.005 0.0001"/>
            <site name="tipsite_{side}" pos="0 0 0" size="0.001"/>
          </body>
        </body>
      </body>"""
    return f"""
<mujoco model="microbot_v1">
  <compiler angle="degree"/>
  <option timestep="0.0005" gravity="0 0 -9.81" cone="elliptic" impratio="10" iterations="60" integrator="implicitfast"/>
  <asset>
    <mesh name="turret" file="{STL}/part_turret.stl" scale="{mm}"/>
    <mesh name="arm" file="{STL}/part_arm.stl" scale="{mm}"/>
    <mesh name="tip" file="{STL}/part_tip.stl" scale="{mm}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" condim="6" friction="{friction} 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>
    <body name="bot" pos="0 0 {BODY_R/1000 + 0.0002}">
      <freejoint name="free"/>
      <geom name="ball" type="sphere" size="{BODY_R/1000}" condim="6" mass="{MASS['body']}" friction="{friction} 0.005 0.0001" rgba="0.3 0.5 0.9 1"/>
      {arm('top', '1 0 0 0')}
      {arm('bot', '0 1 0 0')}
    </body>
  </worldbody>
  <actuator>
""" + "\n".join(f'    <position name="{j}" joint="{j}" kp="0.1" kv="0.004" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>' for j in JOINTS) + """
  </actuator>
</mujoco>"""


class BotSim:
    def __init__(self, friction=0.6, heading_deg=0.0):
        self.model = mujoco.MjModel.from_xml_string(_xml(friction))
        self.data = mujoco.MjData(self.model)
        self.target = dict.fromkeys(JOINTS, 0.0)       # degrees, in CAD convention (pitch positive tilts toward +Y)
        # rest pose: poles horizontal (body +Z along world +X), body +Y pointing down so a positive pitch plants the tip
        R = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
        Rz = np.array([[math.cos(math.radians(heading_deg)), -math.sin(math.radians(heading_deg)), 0],
                       [math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg)), 0], [0, 0, 1]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (Rz @ R).ravel())
        self.data.qpos[3:7] = quat
        self.data.qpos[2] = BODY_R / 1000 + 0.0002
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0

    # ---- control -----------------------------------------------------------------
    def goal(self, **angles):
        """Set joint goals in degrees: az_top, pitch_top, roll_top, az_bot, pitch_bot, roll_bot."""
        for k, v in angles.items():
            self.target[k] = float(np.clip(v, -100, 100)) if k.startswith("pitch") else float(v)

    def _ctrl(self, cmd):
        # MuJoCo hinge about +X by q is Rx(q); the CAD pitch is Rx(-pitch)
        for i, j in enumerate(JOINTS):
            self.data.ctrl[i] = math.radians(-cmd[j] if j.startswith("pitch") else cmd[j])

    def run(self, seconds, log_every=None):
        """Advance, slewing each joint command toward its goal at 120 deg/s. Returns a list of snapshots if log_every."""
        dt = self.model.opt.timestep
        steps = int(round(seconds / dt))
        if not hasattr(self, "cmd"):
            self.cmd = dict(self.target)
        out = []
        for n in range(steps):
            for j in JOINTS:
                err = self.target[j] - self.cmd[j]
                self.cmd[j] += max(-JOINT_SPEED * dt, min(JOINT_SPEED * dt, err))
            self._ctrl(self.cmd)
            mujoco.mj_step(self.model, self.data)
            self.t += dt
            if log_every and n % int(round(log_every / dt)) == 0:
                out.append(self.snapshot())
        return out

    # ---- measurements (mm) -------------------------------------------------------------
    def body_xy(self):
        return self.data.xpos[self.model.body("bot").id][:2] * 1000.0

    def body_z(self):
        return self.data.xpos[self.model.body("bot").id][2] * 1000.0

    def tip_xyz(self, side):
        return self.data.site_xpos[self.model.site(f"tipsite_{side}").id] * 1000.0

    def snapshot(self):
        return {"t": self.t, "body": self.body_xy().copy(), "z": self.body_z(), "tip_top": self.tip_xyz("top").copy(),
                "tip_bot": self.tip_xyz("bot").copy()}

    def contacts_with_floor(self):
        names = set()
        for c in self.data.contact[: self.data.ncon]:
            a, b = self.model.geom(c.geom1).name, self.model.geom(c.geom2).name
            body_a, body_b = self.model.body(self.model.geom_bodyid[c.geom1]).name, self.model.body(self.model.geom_bodyid[c.geom2]).name
            if "floor" in (a, b):
                names.add(body_b if a == "floor" else body_a)
        return sorted(names)

    def up_vector_body_y(self):
        """World z of the body +Y axis: -1 means +Y points straight down (the pitch plane is vertical)."""
        return self.data.xmat[self.model.body("bot").id].reshape(3, 3)[2, 1]


if __name__ == "__main__":
    sim = BotSim()
    sim.run(1.0)
    print("rest: body z %.2f mm (ball radius 15), floor contacts %s, drift %.3f mm" % (sim.body_z(), sim.contacts_with_floor(), np.linalg.norm(sim.body_xy())))
