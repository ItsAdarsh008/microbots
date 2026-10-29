# A trained local policy for the linking swarm

The first learned version of the linking rules (textbook: [Linking 2](../../textbook/L2-from-rules-to-a-model.md)).

**Result:** one small shared network (262 → 128 → 128 → 19, ReLU, 52,627 weights), run by every bot and using only local information, builds every shape every time, including shapes it never trained on. Nothing in the environment is global any more except the teacher's crawl map: which cells are open is decided cell by cell (the readiness rule), and "will anything fall?" is checked with a local holder tree. The structure stays attached at every step.

Python simulation (unseen seeds 100–104, local fall check):

| Shape | Rules (teacher) | Trained network |
| --- | --- | --- |
| Wall (trained on) | 5/5, 50–66 steps | **5/5**, 52–94 steps |
| Arch (trained on) | 5/5, 97–151 steps | **5/5**, 96–116 steps |
| Tower (**never seen**) | 5/5, 60–68 steps | **5/5**, 64–85 steps |

With the old whole-structure fall check instead (`LINK_SAFETY=global`), same weights: wall 48–52, arch 90–98, tower 58–62 steps, also 15/15. The local check costs some speed because it is more cautious.

![Training curve](../../media/linking_training.png)

Training curve (`plot_training.py`, from `policy_log.json`): loss per batch, and how often the network picks the rules' move on held-out decisions. In the DAgger rounds (shaded) new, harder data arrives, so the loss jumps up and agreement on the rules' own runs dips; the network is learning to recover from its own mistakes.

Browser sandbox ("Moves decided by: Trained model"), same weights: see the table in [`EXPERIMENTS.md`](../EXPERIMENTS.md#8-local-layer-choice-local-fall-check-broken-bots-arm-jam), section 8.

## What each bot sees (262 numbers)

- 125: which cells in its 5×5×5 block hold a bot (the floor counts as solid)
- 125: which cells in that block belong to the target shape
- 7: a "how far to the work" signal at its own cell and its 6 face neighbours, relayed between bots up to 2 cells apart (like Kilobot infrared)
- 1: whether it is locked into the shape
- 4: a compass: the direction to the nearest open part of the shape, worked out from the shape map and the bot's own position (the Kilobots had both)

No bot ever sees the whole grid.

## What it outputs

19 scores: stay, 6 slides, 12 pivots. Moves that aren't legal right now are masked out. The bot picks one in proportion to the scores. A bot that is holding someone up who has nowhere else to hang from doesn't move at all (the holder tree in `env.py`, textbook Part 8).

## How it was trained

1. **Imitation:** record the hand-written rules on the wall and arch (8 seeds each, ~98,000 decisions, 10% held out to measure agreement) and train the network to predict their move. 10 epochs.
2. **DAgger, 2 rounds:** let the network drive, have the rules label what they would have done at each decision, add those (~208,000 more), retrain 6 epochs each.

A few minutes on a CPU in total. No GPU needed at this size. `dagger.py` writes `policy_log.json` alongside the weights.

## What made the difference

| What the network gets | Wall | Arch | Tower (unseen) | Browser, far build site |
| --- | --- | --- | --- | --- |
| Hop count between **touching** bots only | 5/5 | 2/5 | 3/5 | |
| The teacher's **global** distance map (diagnostic only) | 5/5 | 5/5 | 5/5 | |
| Hop count between bots **up to 2 cells apart** | 5/5 | 5/5 | 5/5 | **0/9** |
| **+ compass** from the map and own position (used) | 5/5 | 5/5 | 5/5 (as fast as the rules) | **9/9** |

1. The global-map diagnostic showed the network was big enough; the touching-only signal was the bottleneck. A 2-cell relay range fixed it.
2. In the browser the build site sits ~9 cells from the pile, too far for the relay. The model had quietly learned "drift this way" from the fixed training layout. Adding the compass fixed it and made it generalise to every sandbox shape.

## Run it

```bash
pip install jax numpy scipy
python linking/learn/evaluate.py           # network vs rules, uses policy.npz
python linking/learn/train_bc.py           # imitation only
python linking/learn/dagger.py --rounds 2  # imitation + DAgger, writes policy.npz
python linking/learn/plot_training.py      # redraw media/linking_training.png from policy_log.json
python linking/learn/export_policy.py      # copy the weights into the browser sandbox
LINK_SAFETY=global python linking/learn/evaluate.py   # same, with the old whole-structure fall check
LINK_SIGNAL=local python linking/learn/dagger.py --out policy_local.npz   # touching-only signal
```

## Not done yet

- **Parking, support pillars and the truss bridge still use rules.** The model builds; spares park by the rules afterwards. It hasn't been trained or tested with real connector strength.
- **It copies the rules; it doesn't beat them.** Reinforcement learning (stage 2) is next, starting with the long bridge under real connector strength.
- **The teacher's crawl map is global.** The network never sees it (it gets the relayed signal and the compass), but the rules it copies were steered by it.

## What the research says about next steps

[`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md): what the papers (mostly 2024 on) say about the next four steps: a random shape generator, the teacher with symmetry augmentation, memory and learned messages, and RL with real connector strength. It ends with a revised plan and a list of citations to double-check. The full research notes behind it are in [`literature_notes/`](literature_notes/).
