# A trained local policy for the linking swarm

The first learned version of the linking rules (textbook: [Linking 2](../../textbook/L2-from-rules-to-a-model.md)).

**Result:** one small shared network (262 → 128 → 128 → 19, ReLU, 52,627 weights), run by every bot and using only local information, builds every shape every time, including shapes it never trained on. The structure stays attached at every step.

Python simulation (unseen seeds 100–104):

| Shape | Rules (teacher) | Trained network |
| --- | --- | --- |
| Wall (trained on) | 5/5, 46–56 steps | **5/5**, 48–72 steps |
| Arch (trained on) | 5/5, 64–70 steps | **5/5**, 65–74 steps |
| Tower (**never seen**) | 5/5, 49–60 steps | **5/5**, 55–58 steps |

Browser sandbox ("Moves decided by: Trained model"), same weights:

| Shapes | Result |
| --- | --- |
| Wall, small arch, tower (cubes view) | 9/9, as fast as the page's rules (small arch faster) |
| Short bridge, table, pyramid, K, long bridge, dome, spiral, curved arch (**never seen**, up to ~900 bots) | **16/16**, slower than the rules on the biggest (e.g. long bridge 139–240 steps) |
| Wall, small arch, tower, short bridge, table in Real render (arm links, **never trained on**) | 10/10 |

## What each bot sees (262 numbers)

- 125: which cells in its 5×5×5 block hold a bot (the floor counts as solid)
- 125: which cells in that block belong to the target shape
- 7: a "how far to the work" signal at its own cell and its 6 face neighbours, relayed between bots up to 2 cells apart (like Kilobot infrared)
- 1: whether it is locked into the shape
- 4: a compass: the direction to the nearest open part of the shape, worked out from the shape map and the bot's own position (the Kilobots had both)

No bot ever sees the whole grid.

## What it outputs

19 scores: stay, 6 slides, 12 pivots. Moves that aren't legal right now are masked out. The bot picks one in proportion to the scores. If that move would leave anything unattached, it tries the next best (this "will anything fall?" check is still global).

## How it was trained

1. **Imitation:** record the hand-written rules on the wall and arch (8 seeds each, ~75,000 decisions) and train the network to predict their move.
2. **DAgger, 2 rounds:** let the network drive, have the rules label what they would have done at each decision, add those (~200,000 more), retrain.

About 6 minutes on a laptop CPU in total. No GPU needed at this size.

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
python linking/learn/export_policy.py      # copy the weights into the browser sandbox
LINK_SIGNAL=local python linking/learn/dagger.py --out policy_local.npz   # touching-only signal
```

## Not done yet

- **Parking, support pillars and the truss bridge still use rules.** The model builds; spares park by the rules afterwards. It hasn't been trained or tested with real connector strength.
- **It copies the rules; it doesn't beat them.** Reinforcement learning (stage 2) is next, starting with the long bridge under real connector strength.
- **The safety check is global.** So is the goal layer: bots are told which layer is being built.
