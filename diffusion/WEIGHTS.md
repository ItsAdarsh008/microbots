# Where the trained weights and datasets are

Not in git (too large). They are in a **private** Hugging Face model repo, `KuanKuanKuan/microbots-flow-swarm` (the owner can grant access or make it public):

- `weights/<run>.pt`: 215 runs, each `{"ema": state_dict, "config": {...}}` (EMA weights only, no optimizer state).
- `data/`: `shapes.npz`, `shapes2.npz`, `shapes3.npz`, `letters.npz`, `emoji_shapes/`.
- `manifest.json`: every run's main configuration and metrics; `README.md`: the model card.

Upload verified on 2026-10-30: every file matched by size and SHA-256 (231 of 231) before the local copies and the rented machines were released.

## Restore
```bash
hf download KuanKuanKuan/microbots-flow-swarm --local-dir hf_models      # needs access
cp -r hf_models/data diffusion/data ; mkdir -p diffusion/demo_models && cp hf_models/weights/<run>.pt diffusion/demo_models/
python3 diffusion/demo/serve.py --port 8000     # needs torch; models listed in demo/models.json
```
The datasets can also be regenerated with `shapes.py` (`--v2`, `--v3`) and `demo/make_letters.py` (deterministic seeds). Training can be rerun from `configs/<run>.json` with `train.py`.
