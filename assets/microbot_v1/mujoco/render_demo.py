"""Render the body demo GIF: two bots side by side, each from a clean rest state.

    MUJOCO_GL=osmesa python render_demo.py out.gif

Left: one roll. The top arm is planted at 30 deg and swept 90 deg in azimuth (the ball rolls about 30 mm perpendicular to its pole axis).
Chaining manoeuvres is NOT shown: a second roll or turn started from the state the first leaves behind does not reproduce the clean-start result. Right: the validated turning gait (nominal rank 3 of
gait_random_best.json), which rotates the line the bot can roll along by about 75 to 85 deg. Both are simulation on the CAD meshes.
"""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from bot import BotSim, JOINT_SPEED
from gait_search import yaw_deg

HERE = Path(__file__).resolve().parent
TURN = json.load(open(HERE / "gait_random_best.json"))[2]["seq"]
W, H, FPS = 360, 270, 20


def script_roll():
    # one clean roll from rest (the measured case: a 90 deg azimuth sweep moves the ball about 30 mm perpendicular to its pole axis)
    return [("roll", dict(pitch_top=30), 0.6), ("roll", dict(az_top=90), 90 / JOINT_SPEED + 0.5), ("hold", dict(), 3.4)]


def script_turn():
    out = []
    state = dict(az_top=0.0, pitch_top=0.0, az_bot=0.0, pitch_bot=0.0)
    for wp in TURN:
        span = max(abs(wp[j] - state[j]) for j in wp)
        out.append(("turn", dict(wp), span / JOINT_SPEED + 0.4)); state = dict(wp)
    out.append(("hold", dict(), 0.8))
    return out


def frames_of(script, label):
    s = BotSim(); s.run(0.3)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.01, 0.0, 0.012); cam.distance = 0.24; cam.azimuth = 90; cam.elevation = -48
    r = mujoco.Renderer(s.model, H, W)
    opt = mujoco.MjvOption()
    dt = s.model.opt.timestep; per = int(round(1.0 / FPS / dt)); out = []; y0 = yaw_deg(s); b0 = s.body_xy().copy()
    for name, goal, dur in script:
        if goal: s.goal(**goal)
        for n in range(int(round(dur / dt))):
            s.run(dt)
            if n % per == 0:
                r.update_scene(s.data, camera=cam, scene_option=opt); img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img); mv = np.linalg.norm(s.body_xy() - b0); yw = ((yaw_deg(s) - y0 + 90) % 180) - 90
                d.rectangle([0, 0, W, 38], fill=(251, 251, 250)); d.text((8, 5), label, fill=(31, 58, 138)); d.text((8, 21), f"{name}   moved {mv:4.0f} mm   line turned {yw:5.0f} deg", fill=(26, 26, 26))
                out.append(img)
    return out


if __name__ == "__main__":
    a = frames_of(script_roll(), "roll along the line (from rest)")
    b = frames_of(script_turn(), "quarter-turn gait (from rest)")
    n = max(len(a), len(b)); a += [a[-1]] * (n - len(a)); b += [b[-1]] * (n - len(b))
    frames = []
    for x, y in zip(a, b):
        im = Image.new("RGB", (2 * W + 4, H), (251, 251, 250)); im.paste(x, (0, 0)); im.paste(y, (W + 4, 0)); frames.append(im.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    frames[0].save(sys.argv[1], save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print("wrote", sys.argv[1], len(frames), "frames")
