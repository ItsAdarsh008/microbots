# Where the run artefacts live (GPU boxes shut down 2026-11-05)

Everything that was on the two A100 boxes (`~/emergent`: runs with checkpoints `ckpt.pt` / `theta.pt`, logs,
`result.json`, gap sweeps `chasm_gen*.json`, gallery GIFs, warm-start inits, queues, worker logs) is in the
**private** Hugging Face model repo `KuanKuanKuan/microbots-flow-swarm` under `emergent/box137/` and
`emergent/box150/` (one folder per box; run names do not collide). Verified file by file against the pulled copy.

    hf download KuanKuanKuan/microbots-flow-swarm --include "emergent/*" --local-dir hf_emergent
    # a single run: --include "emergent/box137/runs/lat6_chasm_cure_warm/*"

Small files (`log.txt`, `result.json`, `chasm_gen*.json`) for every lattice run are also in git under `results/`.
Warm-start files used by `init_from` are in `emergent/box*/init/`.

To resume on new boxes: `./ops.sh sync`, copy `init/` back, `./ops.sh workers HOST 8`, queue configs from
`lat/configs/` (`./ops.sh queue HOST lat/configs/<cfg>.json`). A run restarted with the same name resumes from its
`ckpt.pt` if that file is placed back under `runs/<name>/`.
