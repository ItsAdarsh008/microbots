# microbots
programmable matter, inspired by big hero 6. this repo holds all my work until the final goal. no central brain, hivemind only.

## For collaborators

**Current status:** Phase 0 (setup and foundations) is in progress. The first goal is a 2D random walk of 100 dots, built in NumPy and then JAX, recorded as a short clip. Phases 0–2 are 2D; the swarm moves to 3D from Phase 3. See [ROADMAP.md](ROADMAP.md) for the full plan and per-phase checklists.

**Getting set up (Windows, PowerShell):**

```powershell
git clone https://github.com/ItsAdarsh008/microbots
cd microbots
py install 3.13                 # needs the Python Install Manager from python.org
py -V:3.13 -m venv .venv        # private Python for this project (.venv is gitignored)
.venv\Scripts\activate          # run in every new terminal; prompt shows (.venv)
pip install jax numpy matplotlib pillow
python -c "import jax; print(jax.devices())"   # should list a CPU device
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead. Step 1 of the Phase 0 walkthrough in the roadmap explains what each command does.

## How we work together

The project runs as two parallel lanes, which brings the timeline from about 12 weeks to roughly 7–8:

| Lane | Covers | Phases | Owner |
| --- | --- | --- | --- |
| Shared | JAX foundations, launch | 0, 6 | Both |
| A · Swarm and visuals | Flocking, Kilobot reproduction, 3D viewer, polish, Jev commands | 1, 2, 4, 5 | To be decided |
| B · Learning | CNN warm-up, 2D then 3D self-healing NCA, environment-aware NCA | 3, 3.5 | To be decided |

- **Hand-off between lanes:** the NCA outputs a 32×32×32 grid of filled and empty cells, and the swarm treats the filled cells as its target positions (proposed; to be agreed in week 1).
- **Issues:** one GitHub issue per roadmap checklist item, each with an owner.
- **Branches and pull requests:** nothing merges into `main` without the other person's review.
- **Weekly sync:** a 30-minute call to demo progress, flag blockers and pick the next items.
- **Posting:** each person posts their own lane's progress and tags the other; the launch is posted together.

## Roadmap

The full build plan (phases, collaboration plan, checklists, resources and a beginner walkthrough for Phase 0) is in [ROADMAP.md](ROADMAP.md).
