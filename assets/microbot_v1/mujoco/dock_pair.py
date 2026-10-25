"""Two microbots welded at contact (a stand-in for the docking connectors), rolling in opposite directions: can a docked pair TURN in place?

    MUJOCO_GL=osmesa python dock_pair.py [out.gif]

A single bot rolls ~30 mm along ONE line (perpendicular to its pole axis) and cannot steer without the long turning gait (FINDINGS.md).
Here two bots sit side by side along their pole axes (so both roll along the same direction), 32 mm apart, and a weld equality
constraint holds their bodies at the initial relative pose. Roll one +90 deg and the other -90 deg in azimuth: if the weld holds, the
pair should yaw about its centre. This is a simulation on the CAD meshes with a rigid weld; the real connector (genderless Hirth face +
electro-permanent magnets), which face docks, and its holding torque are NOT modelled.
"""
import math
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import multibot
from bot import BODY_R, JOINTS, JOINT_SPEED

W, H, FPS = 640, 480, 20


def build(off, weld):
    pos = [(-off[0] / 2000.0, -off[1] / 2000.0), (off[0] / 2000.0, off[1] / 2000.0)]      # off = (dx, dy) in mm between the two bodies; x is the pole axis
    x = multibot.xml(2, pos)
    R0 = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R0.ravel())
    for i in range(2):
        x = x.replace(f'<body name="b{i}_bot" pos="0 0 0">', f'<body name="b{i}_bot" pos="{pos[i][0]} {pos[i][1]} {BODY_R/1000 + 0.0002}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">')
    if weld == "connect":      # a ball joint at the touching tips: the pair can pivot about that point (a pivoting dock), unlike a rigid weld
        a = off[0] / 2000.0
        x = x.replace("</mujoco>", f'<equality><connect body1="b0_bot" body2="b1_bot" anchor="0 0 {a}" solref="0.002 1"/></equality></mujoco>')
    elif weld:
        x = x.replace("</mujoco>", '<equality><weld body1="b0_bot" body2="b1_bot" solref="0.002 1"/></equality></mujoco>')
    return x


def yaw_deg(model, data):
    a = data.xpos[model.body("b0_bot").id][:2]; b = data.xpos[model.body("b1_bot").id][:2]
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def inter_bot_contacts(model, data):
    n = 0
    for c in data.contact[: data.ncon]:
        a = model.body(model.geom_bodyid[c.geom1]).name[:2]; b = model.body(model.geom_bodyid[c.geom2]).name[:2]
        n += a != b and a.startswith('b') and b.startswith('b') and not (model.body(model.geom_bodyid[c.geom1]).name == 'world')
    return n


def run(weld=True, signs=(1, -1), off=(22.0, 22.0), out=None):
    model = mujoco.MjModel.from_xml_string(build(off, weld)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data); start_contacts = inter_bot_contacts(model, data)
    dt = model.opt.timestep
    target = [dict.fromkeys(JOINTS, 0.0) for _ in range(2)]; cmd = [dict(t) for t in target]
    script = [(0.5, {"pitch_top": 30}), (90 / JOINT_SPEED + 0.5, {"az_top": 90}), (1.5, {})]
    y0 = yaw_deg(model, data); c0 = np.mean([data.xpos[model.body(f"b{i}_bot").id][:2] for i in range(2)], axis=0)
    frames = []
    if out:
        r = mujoco.Renderer(model, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0, 0, 0); cam.distance = 0.22; cam.azimuth = 90; cam.elevation = -62
    per = int(round(1.0 / FPS / dt)); step = 0
    for dur, goal in script:
        for i in range(2):
            for k, v in goal.items():
                target[i][k] = float(v * signs[i]) if k == "az_top" else float(v)
        for _ in range(int(round(dur / dt))):
            for i in range(2):
                for j_idx, j in enumerate(JOINTS):
                    err = target[i][j] - cmd[i][j]; cmd[i][j] += max(-JOINT_SPEED * dt, min(JOINT_SPEED * dt, err))
                    data.ctrl[i * 6 + j_idx] = math.radians(-cmd[i][j] if j.startswith("pitch") else cmd[i][j])
            mujoco.mj_step(model, data); step += 1
            if out and step % per == 0:
                r.update_scene(data, camera=cam); img = Image.fromarray(r.render()); d = ImageDraw.Draw(img)
                d.rectangle([0, 0, W, 22], fill=(251, 251, 250))
                d.text((8, 5), f"{('pivot dock' if weld == 'connect' else 'docked (welded)') if weld else 'free (no weld)'} pair   pair yaw {((yaw_deg(model, data) - y0 + 180) % 360) - 180:5.1f} deg", fill=(26, 26, 26)); frames.append(img)
    c1 = np.mean([data.xpos[model.body(f"b{i}_bot").id][:2] for i in range(2)], axis=0)
    yaw = ((yaw_deg(model, data) - y0 + 180) % 360) - 180
    zs = [float(data.xmat[model.body(f"b{i}_bot").id].reshape(3, 3)[2, 1]) for i in range(2)]
    if out:
        frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    return {"start_inter_bot_contacts": start_contacts, "yaw_deg": round(float(yaw), 1), "centre_moved_mm": round(float(np.linalg.norm(c1 - c0) * 1000), 1), "bodyY_z": [round(z, 2) for z in zs], "ncon": int(data.ncon)}


if __name__ == "__main__":
    # offsets in mm between the two bodies; x is the pole axis. 82 mm = tips touching end to end (each bot reaches 40.9 mm along its pole)
    for off in [(0.0, 32.0), (22.0, 22.0), (82.0, 0.0)]:
        print("offset", off)
        print("  welded, opposite rolls :", run(True, (1, -1), off))
        print("  welded, same-direction :", run(True, (1, 1), off))
        print("  free,   opposite rolls :", run(False, (1, -1), off))
        print("  free,   same-direction :", run(False, (1, 1), off))
    if len(sys.argv) > 2:
        print("gif:", run(True, (1, -1), tuple(float(v) for v in sys.argv[2].split(",")), out=sys.argv[1]))
