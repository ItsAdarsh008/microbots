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

## Questions to settle together

1. **Bot geometry:** does the CAD bot have six connecting faces? Can it pivot (like M-Blocks) or only slide?
2. **Strength:** how far can a linked arm stick out before the joints fail? In the arch repair, half the bridge sticks out sideways with nothing under it.
3. **Grid or not:** can the NPA output be snapped to a grid without losing the regrowth behaviour?
4. **Which option to try first:** A is quick to try; B is the real research question.
