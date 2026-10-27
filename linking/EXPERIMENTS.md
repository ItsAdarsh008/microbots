# Linking: what was tried, what worked, what didn't

A log of the linking side track, in order, written for Kuan. The sandbox is `linking/sandbox/index.html` (one file, open it in a browser). The first Python model is `linking/lattice.py`. Every change has its own commit with the numbers in the message (`git log -- linking/`).

Step counts are simulation steps to finish a build, from headless test runs (`linking/sandbox/tests/`). They vary run to run because moves are random.

## 1. Python lattice model (`linking/lattice.py`)

Cube bots on a 3D grid, linked face to face. Two moves: **slide** along a surface, **pivot** (half-turn over the edge of the bot underneath). A move is refused if the corner it swings through is full, or if any bot would lose its chain of links to the floor.

| Tried | Result |
| --- | --- |
| Distance field through all empty cells | **Failed.** Spares stacked into ladders trying to climb through the air. |
| Paths only along fixed surfaces (face neighbours) | **Failed.** The top of a 1-bot-thick wall is only reachable by a diagonal pivot, so every cell read "unreachable" and nothing moved. |
| Paths along surfaces with 18-neighbour (diagonal) steps | **Worked.** Sheet, arch and tower all build. |
| Moves applied one at a time | **Bug.** 639 pivots swung through a cell another bot had just left that step. Fixed by only allowing cells that were empty at the start of the step (`busy`). |
| Cuts with no floor anchor | **Failed.** Cutting the arch's bridge dropped a whole leg (114 bots instead of 24). Fixed: the floor counts as a fixed bot. |
| Spares left where they finished | **Failed.** They sat on top of the shape, so cutting the top took them too. Fixed: spares park on the floor before damage. |

Result: 17/17 checks in `linking/checks.py` (3 shapes x 3 seeds, connectivity every step, no overlaps, locked bots never move, two repairs).

## 2. In-browser sandbox, cube model

Same rules ported to JavaScript with a three.js view.

| Tried | Result |
| --- | --- |
| Parking: any empty floor cell | **Slow.** Spares wandered for ~120 steps. Fixed with an ordered pad (far column first, centre out, lock on arrival): ~45 steps. |
| v1 strength limit ("a bot with nothing under it must touch a bot that has something under it"), no planning | **Unreliable.** Spares sometimes stacked under the long deck as scaffolding by accident and finished; other runs stuck at ~93%. |
| Support planner: add columns under any cell that would overhang more than one bot | **Worked.** Long bridge needed 45 support bots (later 245 at the bigger size). Every shape builds with the limit on. |
| Crawl map ignoring the strength limit | **Failed.** Bots kept trying to climb down off deck edges, which the limit refuses. Fixed: map avoids cells the limit would refuse. |
| Unjam: after 3+ steps with no progress, allow backward moves | **Worked.** Untangled knots of spares hanging off support columns. |
| Cut with limit on, no extra drops | **Failed.** Deck ends left sticking out made every later move illegal. Fixed: a cut also drops anything left overhanging. |

## 3. Scaling to ~1,000 bots

| Tried | Result |
| --- | --- |
| Full connectivity flood after every move | Too slow at 1,000 bots. |
| Local check in the 3x3x3 block around the bot that moved, full check only if unsure | **Worked**, 4 to 23 ms per step. **First version had a bug**: a bot on the floor with one bot above it was treated as holding nobody up (the floor wasn't counted as a neighbour). Caught by checking connectivity every step in tests. |
| Dome without keep-out | **Failed** at 97%: 120 bots crawled inside and got sealed in. Fixed: cells the finished shape will enclose are off limits (skipped when supports stand inside). |

## 4. Splitting the swarm between build sites

| Tried | Result |
| --- | --- |
| Everyone heads for the nearest empty cell | The swarm moves as one mass to the nearest site. |
| Group goal cells into sites, give each its own distance map, share bots by work left | **Splits correctly.** Faster for the table (86 vs ~100 to 115 steps), slower for the bridge (225 vs ~170) and arch (298 vs ~185), because half the bots walk to the far tower. |
| Strict fill order (most-enclosed cells first) + max 2 bots per cell + unclaimed bots wait | **Failed badly.** Most shapes stalled at 13 to 97%: the strict order picked cells nobody could reach yet, and waiting bots formed a wall in front of the work. Reverted. |
| Claim only holes (cells filled on 3+ sides) with one bot each + others give way within 2 cells; drop claims on stalls | **Worked.** Everything builds again. |
| Crawl map counting any diagonal as one move | **Bug** behind deck holes that never filled: bots crowded under holes they couldn't enter. A diagonal step is only possible if one of the two corner cells is free. Fixed: spiral ~299 to ~171 steps; a table case stuck at 99.5% now finishes 8/8. |

## 5. Arm model (after microbot v1)

Each bot points its two arms along one axis (x, y or z). A link is one bot's arm tip in a side port of the neighbour, so the neighbour's arms must point another way. Settings are in the `ARMS` block at the top of the script (55.5 mm step, 30 mm ball, tip-to-tip and ball-to-ball links off because 81 mm and 28 mm don't fit the grid).

| Tried | Result |
| --- | --- |
| Arm model, local check in 3x3x3 | Works but 30 to 88 ms per step (local check often unsure). |
| Local check in 5x5x5 + skip arm directions that link to nothing | **Worked**, 7 to 37 ms per step. All 8 shapes build. |
| Arm model + strength limit | **Not solved.** Builds stall at 83 to 99%: bots under the deck hang off each other in chains and none can let go first. |
| Arm direction: keep the old one if it works | Arms pointed at random, sticking out of pillars. |
| Arm direction: prefer both arms pointing at bots or shape cells, ties to vertical | **Worked.** Pillars alternate vertical holders and horizontal held bots; decks run arms along the deck. |
| Each arm aimed independently (pole + two arms, re-aim free arms when a neighbour arrives) | **Failed.** Arms ran out (528 of 600 plugged in) and arriving bots often had nothing free to plug into; every shape stalled. Reverted. Free arms are now only *drawn* folded against a neighbour. |

## 6. Triangle truss bridge

Separate grid of triangles (6 neighbours at 60 degrees). Bots link to every occupied neighbour, move by rolling 60 degrees around a shared neighbour, and bend two arms toward neighbours so arm pairs meet tip to tip.

| Tried | Result |
| --- | --- |
| "Done" = every target node occupied | **Bug.** Bots passing through counted as filled, then walked off. Fixed by locking. 5/5 builds then finish in ~200 steps. |
| Parking with paths over loose bots | Spares made dangling chains under the span. Fixed: parking paths only along fixed bots. A couple of spares can still end up hanging under the span. |
| Cut 12 bots from the span | Rebuilt in ~35 to 45 steps. |

## Still open

1. **Moves aren't real v1 moves.** Bots slide and pivot (or roll, on the triangle grid). v1 moves with two 3-joint arms. Needed: how does v1 actually get from one spot to the next?
2. **Arm model + strength limit jams** near the end of every build.
3. **Not local yet.** The crawl directions, the work sharing, the support planner and the "will anything fall" check all look at the whole build. Real bots only see their neighbours. This is where a learned policy (NPA-style) could come in.
4. **Numbers to confirm:** 3 N per connector, whether tip-to-tip docking is possible, and at what spacing.
