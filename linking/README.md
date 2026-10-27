# Linking: bots that stay attached while they move

Kilobots never attached to each other; they just stopped next to each other on a table. Real 3D programmable matter has to hold itself up, so every bot must stay **linked** to the structure while the swarm reshapes. This folder is a first model of that.

![The two moves and the two rules](../media/linking_moves.png)

## The model

- **Bots are cubes on a 3D grid.** Two bots that share a face are linked. The floor counts as one big bot that never moves.
- **Two moves only**, both keep the bot touching something the whole time:
  - **slide:** glide one cell along a surface
  - **pivot:** roll half a turn over the edge of the bot underneath, from its top face onto its side face. This is how MIT's M-Blocks move.
- **Two rules:**
  - A pivot needs the corner cell it swings through to be empty.
  - No move may leave any bot without a chain of links to the floor. Nothing ever falls.
- **All bots move at the same time.** A bot may only move into cells that were empty when the step began and that no other bot has claimed.
- **Building:** the target is filled bottom layer first. Free bots crawl over the outside of the structure towards the empty target cells and lock in place when they arrive. Spare bots then climb down and park on the floor.
- **Repair:** cut part of the finished shape away. Anything no longer linked to the floor falls off. The spares crawl back up and rebuild.

## Clips

| Clip | What it shows |
| --- | --- |
| `media/captures/linking_sheet.mp4` | A one-bot-thick wall (81 cells), the "surface" test |
| `media/captures/linking_tower.mp4` | A hollow square tower (160 cells) |
| `media/captures/linking_arch_repair.mp4` | An arch is built, the middle of its bridge is cut out, and the swarm rebuilds it. Half the bridge hangs on as a cantilever because it's linked. |
| `media/captures/linking_sheet_repair.mp4` | The top of the wall is cut off and regrown |

Colours: purple = locked into the shape, grey = free, orange = pivoting, red = removed by damage. Teal bars are links.

## Running it

```powershell
pip install -r requirements.txt          # adds scipy
python linking/checks.py                 # should end with 17/17 checks passed (about 30 seconds)
python linking/animate.py --shape arch   # also: sheet, tower; add --repair for the cut-and-regrow demo
python linking/animate.py --shape arch --repair --save media/captures/linking_arch_repair.mp4
python linking/diagram.py                # redraws media/linking_moves.png
```

Textbook: [Linking chapter](../textbook/L1-linking.md).

## What is not local yet

Two parts are computed over the whole grid every step:

1. **The distance field** that tells each bot which way to crawl. A real swarm would spread it as a hop-count gradient through the links, like Phase 2.
2. **The "will anything fall?" check.** A real bot can only see its neighbours, so it needs a local test for whether it's safe to let go.

Both were kept global on purpose, so the move rules could be tested on their own. Replacing them with something local is the open problem, and it's where the learned policy could come in.

## Combining this with the NPA work

The NPA work answers **where** bots should be: a shape that regrows after cuts. This folder answers **how** linked bots can get there without falling apart. Three ways to join them, from easiest to most ambitious:

| | Idea | What it needs |
| --- | --- | --- |
| A | **Hand-off grid.** The NPA's particle positions are turned into a target grid of filled cells. The lattice swarm fills it. When the NPA regrows a cut, the new cells become new targets. | A way to turn NPA particles into a voxel grid (the 32×32×32 hand-off we already proposed) |
| B | **Masked action policy.** Train an NPA-style shared MLP whose output is a choice between *stay*, the slides and the pivots. Moves that break the two rules are masked out, so the policy can never drop a bot. It sees only its 3×3×3 neighbourhood plus its own hidden state. | A training setup on the lattice; the policy would replace the global distance field |
| C | **Free particles that snap.** Keep the NPA's free movement, but particles latch when they touch and may only unlatch if nothing falls. | Mostly a comparison: shows what the free model gets away with |

## Checked against the microbot v1 design

[`assets/microbot_v1/DESIGN.md`](../assets/microbot_v1/DESIGN.md) answers some of the open questions:

| Question | What v1 says | What it means for this model |
| --- | --- | --- |
| How many links per bot? | 6 genderless connectors (2 arm tips, 4 ports), and `block_2x2x2` uses all 6 on a grid | Matches the 6-face cube model here |
| How does a bot move? | With two 3-joint arms, not by rolling like an M-Block | "Slide" and "pivot" are stand-ins for "move to a neighbouring or diagonal cell while staying linked". Each needs a matching arm sequence on the real bot. |
| How strong are links? | About 3 N per connector (assumed); only **one** bot can stick out sideways before a connector peels | The arch repair breaks this: half the bridge sticks out several bots long. The model needs a cantilever limit. |
| Grid or not? | The 32³ target can map onto the `block_2x2x2` lattice (28 × 28 × 81 mm cells) or onto triangulated 3-bot cells | The hand-off grid (option A) fits v1 directly |

## Questions to settle together

1. **Moves:** which arm sequences give a slide and a pivot on v1? Is there a move v1 can do that the lattice model is missing?
2. **Cantilever limit:** should rule 2 become "nothing falls **and** nothing sticks out more than one bot sideways"? That changes build order: bridges and overhangs would need temporary supports, or triangulated cells.
3. **Lattice:** cubic (`block_2x2x2`) or triangulated? The cube model here only covers cubic.
4. **Which option to try first:** A is quick to try; B is the real research question.

## For Kuan: what worked and what didn't

- [`EXPERIMENTS.md`](EXPERIMENTS.md): every approach tried, in order, with the numbers and what failed
- [`sandbox/index.html`](sandbox/index.html): the in-browser sandbox (one file)
- [`sandbox/tests/`](sandbox/tests/): headless runs that produce the step counts
- `git log -- linking/`: one commit per change, with test results in the message
- [`AGENT_TRACE.md`](AGENT_TRACE.md): the AI coding session that built all this (requests, commands, results), cleaned up
