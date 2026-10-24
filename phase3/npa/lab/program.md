# Ground rules for experiment agents (autoresearch-style)

You are exploring which per-particle networks and training schemes make the lizard swarm repair itself best.
The loop: propose one change, queue it, read the score, keep what helps, write down what you learned.

**You may add or edit:** `lab/cells*.py` (networks), `lab/trainer_*.py` (training schemes), config files in
`lab/queue/pending/`, and notes. Prefer changing one thing at a time, and prefer the simpler code on ties.

**You must not edit:** `npa.py`, `target.py`, `lab/evaluate.py`, `lab/damage.py`, `data/` or anything else
hashed by `evaluate.fingerprint()`. Every result row records those hashes; a row whose hashes differ from the
baseline's is invalid. Do not look for ways to make the score better without making the swarm repair better
(changing seeds or severities, skipping training steps, shrinking the lizard, special-casing the evaluator).

**Score:** `summary.score` in `results.jsonl` is the mean loss after damage and recovery over six damage
kinds (lower is better). It includes growth quality; `summary.excess` is the part due to damage. Always look at
the per-kind numbers, `summary.drift` (does the finished lizard stay put?), and at least one picture before
believing a win. A gain smaller than the spread between seeds is not a gain: confirm candidates with seeds
1 and 2 before claiming anything.

**Training damage vs evaluation damage:** the evaluator uses fixed held-out seeds and severities, but it does
apply all six kinds. A config that trains on a kind is no longer held out for that kind; say so when reporting.

**Stopping:** create `lab/STOP` to stop the queue runner.
