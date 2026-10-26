# NPA lab release (2026-10-26): results, weights and notes

What this is: the data behind `phase3/npa/lab/FINDINGS.md`, from a project that trains small shared neural controllers so a swarm of simulated bots keeps or regrows a shape after damage. **Read the limits first.**

## Limits (please read)
- **All learned results use about 1,000 bots (1,024) with about 7 neighbours each.** Nothing was run or evaluated below roughly 1,000 bots. Whether any of it shows up with 3 to 10 bodies is untested, and on 2026-10-25 the project owner decided the swarm training is probably not relevant for a handful of bodies, so the GPU training was stopped. Treat this as an archive of what was tried, not as a method that works for small robot groups.
- **The "body model" is an assumption.** `lab/embody.py` is a statistical stand-in (one rolling line per bot, noisy sensing, packet loss, random headings). Its motion limits come from MuJoCo simulation on a CAD model of the bot (`assets/microbot_v1/mujoco/`); nothing has been built or measured on hardware.
- **Most numbers are single runs.** Where a second seed exists it is named `_s1` or `_s2`. Several earlier claims were corrected later; see the `note` column and the correction sections in `FINDINGS.md`.
- **The second sensing ring (scales [1, 2]) is an assumption.** A line-of-sight count (`lab/los_check.py`) suggests only about a quarter of its bots would be optically visible.
- **Damage kinds:** the evaluator also deletes bots ("cut"). The owner has said deletion is not a feasible real-world event, so cut results are of limited use; "drag" and "clump" (bots displaced) are the closer cases.

## Files
| Path | What |
| --- | --- |
| `results.csv` | one row per finished run (404): configuration flags, final scores, and for the shape-agnostic runs the **gain over doing nothing** per damage kind and severity (`gain_<kind>_<small|medium|large>`, higher is better, 0 = no better than doing nothing, negative = worse) |
| `summaries.json` | the full config and summary of every run |
| `weights/<run>.npz` and `.config.json` | trained weights (NumPy) of the 19 runs the findings rely on, with their configs. Plain cells store `w1`, `b1`, `w2`, `eps`; other cells store more tensors. Load the model with `phase3/npa/lab/cells.py` (`cells.build(**config["cell"])`) |
| `../lab/FINDINGS.md`, `hypotheses.md`, `robot_hypotheses.md`, `ROBOT_RESEARCH.md`, `LITERATURE_selfrepair.md` | the write-up and the literature notes (the robot research file is model-written; its sources are marked read or snippet) |
| `../../../assets/microbot_v1/mujoco/` | the MuJoCo bot model, gait searches, docked-pair tests and their `FINDINGS.md` |
| `../../../media/`, `../dashboard/gifs/` | the clips (swarm runs, free vs one-line comparison, six CAD bots in one world) |

## Units and conventions
- 1 world unit = 1,160 mm; body diameter 30 mm (0.0259 units); sensing radius eps 0.05 (58 mm).
- Scores are losses (lower is better): density + 5 x colour against the start swarm. `control_loss` is the loss of the undamaged swarm; a good model has `score` close to it.
- Per-severity gains are means over 25 held-out shapes (5 emoji, 20 procedural) and 2 samples each.

## Not included (on purpose)
- The paper authors' pretrained lizard weights (no licence; git-ignored).
- The CAD meshes (they live on a separate branch of the owner's repository; permission to redistribute not checked).
- Per-run evaluation arrays (about 75 MB; regenerate with `lab/run.py` or read `runs/*/result.json` locally).

## Rebuild this folder
`python phase3/npa/lab/make_release.py` (needs `runs/` locally).
