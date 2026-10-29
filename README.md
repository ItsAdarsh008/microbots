# microbots
programmable matter, inspired by big hero 6. this repo holds all my work until the final goal. no central brain, hivemind only.

**See [progress.md](progress.md)** for pictures and animations of every step so far, from the first random dots to linked 3D bots.

## For collaborators

**Current status:** Phases 0–2 are built. Phase 0 is a 2D random walk in NumPy and JAX ([Chapter 0](textbook/00-random-walk.md)); Phase 1 is a 1,000-bot flock that uses only local rules ([Chapter 1](textbook/01-flocking.md)); Phase 2 recreates the 2014 Kilobot self-assembly algorithm, with 150 robots building a star, a wall or the letter K ([Chapter 2](textbook/02-self-assembly.md)). **Linking** (`linking/`) is a side track: 3D cube bots that stay attached while they slide and pivot over each other, build a wall, tower or arch, and repair it after damage ([Linking chapter](textbook/L1-linking.md), [discussion notes](linking/README.md)). Next: clear the Chapter 0–2 understanding gates, then Phase 3 (the self-healing neural cellular automaton, moving to 3D). Phases 0–2 are 2D; the swarm moves to 3D from Phase 3. See [ROADMAP.md](ROADMAP.md) for the full plan and per-phase checklists.

![150 robots assembling themselves into the letter K](media/assemble_k.gif)

**What's in the repo:**

| Path | What it is |
| --- | --- |
| `phase0/` | Phase 0 code: NumPy and JAX random walks, `animate.py`, `checks.py` |
| `phase1/` | Phase 1 code: local-rule flocking with brute-force and cell-list neighbour search, `animate.py`, `checks.py` |
| `phase2/` | Phase 2 code: Kilobot-style self-assembly (gradient, trilateration, edge-following), `animate.py`, `checks.py` |
| `assets/microbot_v1/` | The base microbot: parametric CadQuery model, design note and an in-browser physics sandbox |
| `linking/` | Linked 3D cube bots: slide and pivot moves, build and repair, `animate.py`, `checks.py`, `diagram.py` |
| `textbook/` | One chapter per phase explaining the code from scratch, with exercises |
| `media/` | Rendered clips and GIFs |
| `ROADMAP.md` | The full plan, timeline and checklists |

Code is written with AI help to move fast; the textbook is how it gets understood. Read a phase's chapter before building on that phase.

**Getting set up (Windows, PowerShell):**

```powershell
git clone https://github.com/ItsAdarsh008/microbots
cd microbots
py install 3.13                 # needs the Python Install Manager from python.org
py -V:3.13 -m venv .venv        # private Python for this project (.venv is gitignored)
.venv\Scripts\activate          # run in every new terminal; prompt shows (.venv)
pip install -r requirements.txt                # jax, numpy, matplotlib, pillow, scipy
python -c "import jax; print(jax.devices())"   # should list a CPU device
```

**Running Phase 0:**

```powershell
python phase0/checks.py                 # should end with 8/8 checks passed
python phase0/animate.py --bots 1000    # opens the animation in a window
python phase0/animate.py --bots 1000 --save media/random_walk.mp4   # MP4 needs ffmpeg
```

**Running Phase 1:**

```powershell
python phase1/checks.py                 # should end with 10/10 checks passed
python phase1/animate.py                # 1,000-bot flock, beacon switches on halfway
```

**Running Phase 2:**

```powershell
python phase2/checks.py                 # should end with 12/12 checks passed (2-3 minutes)
python phase2/animate.py --shape k      # also: --shape star, --shape wall
```

**Running linking:**

```powershell
python linking/checks.py                       # should end with 17/17 checks passed (~30 s)
python linking/animate.py --shape arch --repair   # also: --shape sheet, --shape tower
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
- **Posting:** nothing about the project goes public until it's about 80% done (Phases 3 and 3.5 working, around week 9). Until then, capture every clip into the shared capture folder and keep warming up your own X account with replies and general "today I learned" posts. Teasers start at 80%; the launch is posted together.

## Roadmap

The full build plan (phases, collaboration plan, checklists, resources and a beginner walkthrough for Phase 0) is in [ROADMAP.md](ROADMAP.md).
