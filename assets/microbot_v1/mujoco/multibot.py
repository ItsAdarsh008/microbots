"""Several microbots (default 6) in one MuJoCo world, on the CAD meshes, each doing a roll from rest at its own heading.

    MUJOCO_GL=osmesa python multibot.py out.gif [n_bots]

Same bot model as bot.py (one copy per bot, bodies collide with each other and the floor). This is a movement demo, not
swarm self-repair: no controller reacts to its neighbours, each bot just runs the validated one-roll script
(top arm planted at 30 deg, then a 90 deg azimuth sweep, about 30 mm along the line perpendicular to its pole axis;
bots roll in alternating directions). The toothed face, magnets and cables are not modelled (see bot.py).
"""
import math
import sys
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import bot
from bot import BODY_R, PIN_Z, ARM_LEN, MASS, SERVO_TORQUE, STL, JOINTS, JOINT_SPEED

W, H, FPS = 640, 480, 20


def xml(n, positions, friction=0.6):
    mm = "0.001 0.001 0.001"
    def arm(p, side, quat):
        return f"""
      <body name="{p}turret_{side}" quat="{quat}">
        <joint name="{p}az_{side}" type="hinge" axis="0 0 1" damping="0.002" armature="2e-6"/>
        <geom type="mesh" mesh="turret" mass="{MASS['turret']}" rgba="0.55 0.55 0.6 1" friction="{friction} 0.005 0.0001"/>
        <body name="{p}arm_{side}" pos="0 0 {PIN_Z/1000}">
          <joint name="{p}pitch_{side}" type="hinge" axis="1 0 0" range="-100 100" limited="true" damping="0.002" armature="2e-6"/>
          <geom type="mesh" mesh="arm" pos="0 0 {-PIN_Z/1000}" mass="{MASS['arm']}" rgba="0.8 0.8 0.85 1" friction="{friction} 0.005 0.0001"/>
          <body name="{p}tip_{side}" pos="0 0 {ARM_LEN/1000}">
            <joint name="{p}roll_{side}" type="hinge" axis="0 0 1" damping="0.001" armature="1e-6"/>
            <geom type="mesh" mesh="tip" mass="{MASS['tip']}" rgba="0.9 0.5 0.2 1" friction="{friction} 0.005 0.0001"/>
          </body>
        </body>
      </body>"""
    palette = ["0.3 0.5 0.9 1", "0.9 0.4 0.4 1", "0.3 0.75 0.45 1", "0.85 0.7 0.25 1", "0.6 0.45 0.85 1", "0.3 0.75 0.8 1"]
    bodies = ""
    for i in range(n):
        p = f"b{i}_"
        bodies += f"""
    <body name="{p}bot" pos="0 0 0">
      <freejoint name="{p}free"/>
      <geom name="{p}ball" type="sphere" size="{BODY_R/1000}" condim="6" mass="{MASS['body']}" friction="{friction} 0.005 0.0001" rgba="{palette[i % len(palette)]}"/>
      {arm(p, 'top', '1 0 0 0')}
      {arm(p, 'bot', '0 1 0 0')}
    </body>"""
    acts = "\n".join(f'    <position name="{f"b{i}_"}{j}" joint="{f"b{i}_"}{j}" kp="0.1" kv="0.004" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>' for i in range(n) for j in JOINTS)
    return f"""
<mujoco model="microbots">
  <compiler angle="degree"/>
  <option timestep="0.0005" gravity="0 0 -9.81" cone="elliptic" impratio="10" iterations="60" integrator="implicitfast"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.88 0.86" rgb2="0.74 0.74 0.72" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="100 100"/>
    <mesh name="turret" file="{STL}/part_turret.stl" scale="{mm}"/>
    <mesh name="arm" file="{STL}/part_arm.stl" scale="{mm}"/>
    <mesh name="tip" file="{STL}/part_tip.stl" scale="{mm}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid" condim="6" friction="{friction} 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>
    <light pos="0 0 1" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>{bodies}
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>"""


def main(out, n=6):
    cols = 3
    pos = [((i % cols - (cols - 1) / 2) * 0.075, (i // cols - ((n - 1) // cols) / 2) * 0.075) for i in range(n)]
    heads = [0, 25, -20, 40, -35, 15][:n] if n <= 6 else [(17 * i) % 90 - 45 for i in range(n)]
    model = mujoco.MjModel.from_xml_string(xml(n, pos))
    data = mujoco.MjData(model)
    R0 = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    for i in range(n):
        h = math.radians(heads[i]); Rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
        q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rz @ R0).ravel())
        a = model.joint(f"b{i}_free").qposadr[0]
        data.qpos[a:a + 3] = (pos[i][0], pos[i][1], BODY_R / 1000 + 0.0002); data.qpos[a + 3:a + 7] = q
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    target = [dict.fromkeys(JOINTS, 0.0) for _ in range(n)]
    cmd = [dict(t) for t in target]
    start = [data.xpos[model.body(f"b{i}_bot").id][:2].copy() for i in range(n)]
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0, 0, 0.0); cam.distance = 0.42; cam.azimuth = 90; cam.elevation = -62
    per = int(round(1.0 / FPS / dt))
    script = [(0.5, {"pitch_top": 30}), (90 / JOINT_SPEED + 0.5, {"az_top": 90}), (0.6, {}), (0.6, {"az_top": 0, "pitch_top": 0}), (90 / JOINT_SPEED + 0.8, {"pitch_top": 30}), (90 / JOINT_SPEED + 0.5, {"az_top": -90}), (1.4, {})]
    frames, step, t = [], 0, 0.0
    for dur, goal in script:
        for i in range(n):
            for k, v in goal.items():
                s = 1 if i % 2 == 0 else -1
                target[i][k] = float(v * s) if k == "az_top" else float(v)
        for _ in range(int(round(dur / dt))):
            for i in range(n):
                p = f"b{i}_"
                for j_idx, j in enumerate(JOINTS):
                    err = target[i][j] - cmd[i][j]
                    cmd[i][j] += max(-JOINT_SPEED * dt, min(JOINT_SPEED * dt, err))
                    data.ctrl[i * 6 + j_idx] = math.radians(-cmd[i][j] if j.startswith("pitch") else cmd[i][j])
            mujoco.mj_step(model, data); t += dt; step += 1
            if step % per == 0:
                r.update_scene(data, camera=cam); img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img); d.rectangle([0, 0, W, 22], fill=(251, 251, 250))
                mv = [np.linalg.norm((data.xpos[model.body(f"b{i}_bot").id][:2] - start[i]) * 1000) for i in range(n)]
                d.text((8, 5), f"{n} microbots, CAD meshes, MuJoCo   t = {t:4.1f} s   moved (mm): " + " ".join(f"{m:3.0f}" for m in mv), fill=(26, 26, 26))
                frames.append(img)
    frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    fin = [np.linalg.norm((data.xpos[model.body(f"b{i}_bot").id][:2] - start[i]) * 1000) for i in range(n)]
    up = [float(data.xmat[model.body(f"b{i}_bot").id].reshape(3, 3)[2, 1]) for i in range(n)]
    print("wrote", out, len(frames), "frames; final moved (mm):", [round(float(v)) for v in fin], "body +Y z (-1 = upright):", [round(u, 2) for u in up])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
