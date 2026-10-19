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

## Roadmap

The full build plan (phases, checklists, resources and a beginner walkthrough for Phase 0) is in [ROADMAP.md](ROADMAP.md).
