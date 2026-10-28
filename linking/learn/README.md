# A trained local policy for the linking swarm

The first learned version of the linking rules (textbook: [Linking 2](../../textbook/L2-from-rules-to-a-model.md)).

**Result:** one small shared network (258 → 128 → 128 → 19, ReLU, 52,115 weights), run by every bot and seeing only what is near it, builds every shape every time, including a shape it never trained on. The structure stays attached at every step.

| Shape | Rules (teacher) | Trained network |
| --- | --- | --- |
| Wall (trained on) | 5/5, 46–56 steps | **5/5**, 51–64 steps |
| Arch (trained on) | 5/5, 64–70 steps | **5/5**, 64–79 steps |
| Tower (**never seen in training**) | 5/5, 49–60 steps | **5/5**, 68–135 steps |

Evaluation uses seeds 100–104, none of which appear in training.

## What each bot sees (258 numbers)

- 125: which cells in its 5×5×5 block hold a bot (the floor counts as solid)
- 125: which cells in that block belong to the target shape
- 7: a "how far to the work" signal at its own cell and its 6 face neighbours
- 1: whether it is locked into the shape

The signal is a hop count relayed between bots up to 2 cells apart, like the Kilobots' infrared. No bot ever sees the whole grid.

## What it outputs

19 scores: stay, 6 slides, 12 pivots. Moves that aren't legal right now are masked out. The bot picks one in proportion to the scores. If that move would leave anything unattached, it tries the next best (this "will anything fall?" check is still global).

## How it was trained

1. **Imitation:** record the hand-written rules on the wall and arch (8 seeds each, ~75,000 decisions) and train the network to predict their move.
2. **DAgger, 2 rounds:** let the network drive, have the rules label what they would have done at each decision, add those (~200,000 more), retrain.

About 6 minutes on a laptop CPU in total. No GPU needed at this size.

## What made the difference

| Signal the network gets | Wall | Arch | Tower (unseen) |
| --- | --- | --- | --- |
| Hop count between **touching** bots only | 5/5 | 2/5 | 3/5 |
| Hop count between bots **up to 2 cells apart** (used) | 5/5 | 5/5 | 5/5 |
| The teacher's **global** distance map (diagnostic only) | 5/5 | 5/5 | 5/5 |

The same network with the global map builds everything, so the network was never the problem: the touching-only signal was. A slightly longer relay range closes the gap while staying local.

## Run it

```bash
pip install jax numpy scipy
python linking/learn/evaluate.py           # network vs rules, uses policy.npz
python linking/learn/train_bc.py           # imitation only
python linking/learn/dagger.py --rounds 2  # imitation + DAgger, writes policy.npz
LINK_SIGNAL=local python linking/learn/dagger.py --out policy_local.npz   # touching-only signal
```

## Not done yet

- **Not in the browser sandbox yet.** This runs on the Python simulation (`linking/lattice.py`: wall, arch, tower, cube model). Next: a "Rules / Learned" switch in the sandbox.
- **It copies the rules; it doesn't beat them.** Reinforcement learning (stage 2) is next, starting with the long bridge under real connector strength.
- **The safety check is global.** So is the goal layer: bots are told which layer is being built.
