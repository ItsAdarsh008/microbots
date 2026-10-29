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

## 7. First trained policy (`linking/learn/`)

One shared network (258 -> 128 -> 128 -> 19, 52,115 weights) sees only a 5x5x5 block, the shape map for that block, and a relayed "how far to the work" signal. Masked so only legal moves can be picked. Evaluated on unseen seeds; the tower was never in training.

| Tried | Wall | Arch | Tower (unseen) |
| --- | --- | --- | --- |
| Imitation only, signal between touching bots | 5/5 (62-92 steps) | 2/5 | 3/5 |
| + DAgger, 2 rounds | 5/5 (50-72) | 2/5 | 3/5 |
| + DAgger, teacher's global distance map (diagnostic) | 5/5 | 5/5 | 5/5 |
| **+ DAgger, signal relayed between bots up to 2 cells apart** | **5/5 (51-64)** | **5/5 (64-79)** | **5/5 (68-135)** |

The diagnostic showed the network was big enough and the touching-only signal was the bottleneck. A Kilobot-style relay range of 2 cells fixed it while staying local. Teacher for comparison: 46-56, 64-70, 49-60 steps.

**In the browser** (same weights, observation checked identical to Python on 30 bots) it first failed every build, 0/9: the build site is ~9 cells from the pile, beyond the relay, and the model had learned "drift +x" from the fixed training layout. **Fix:** a 4-number compass (direction and closeness to the nearest open shape cell, from the map and the bot's own position, as Kilobots had). Retrained: Python 15/15 (tower now as fast as the rules); browser 9/9 on the training shapes, **16/16 on eight bigger unseen shapes** (up to ~900 bots), and 10/10 in Real render (arm links, never trained on).

## 8. Local layer choice, local fall check, broken bots, arm jam

Step counts from `sandbox/tests/run_shapes.js` (one run each) and `learn/evaluate.py`.

**Readiness rule** (replaces "lowest unfinished layer"). A shape cell opens when it's on the floor, the shape cell under it is locked, or (overhang) a locked bot is beside it; and every shape cell in the layer below within `LEVEL_REACH` cells is locked.

| Tried | Result |
| --- | --- |
| Only "cell under is filled" | Builds, but columns race ahead. Python arch 86–93 steps (was 64–70): one leg and the deck go up first, then everyone walks back to finish the other leg. |
| + wait for the 4 cells diagonally below | Wall back to normal speed; arch still ~90. |
| Strength limit, overhang opens beside any locked bot | **Bug:** dome stuck at 73%, K took 1354 steps. Fixed: with the limit on, the bot beside must itself stand on something. |
| + wait for the layer below within 2 cells | Dome under the strength limit failed about half the runs: walls rose and sealed off the inside pillars. |
| **Within 3 cells (used)** | Dome 6/6. With the old global fall check, most browser shapes built faster than with one global layer (long bridge ~125 steps vs 170–225). |

**Holder tree** (replaces the global "will anything fall?" flood). Every bot has a holder; a bot may move only if each bot it holds can switch to a neighbour whose chain reaches an anchor (floor or locked bot) within 16 hops without passing through it.

| Tried | Result |
| --- | --- |
| Depth numbers (switch only to a lower neighbour) | Safe, but the arch stuck: free scaffold bots became holders of locked deck bots and could never leave. Fixed: locked bots hang only from locked bots. |
| Ask up the chain instead of depth numbers | **Bug:** a bot whose neighbours all had long chains got no holder, so nobody knew it depended on them, and it fell (caught by the every-step connectivity test). Fixed: a moved bot always takes some holder. |
| + hand held bots over one at a time | Fewer bots stuck holding someone. |
| + "stalled" means nothing locked in (not "nothing moved"), so unjam kicks in | Arm-model dome, spiral, arch finish again (they had been shuffling without progress). Python teacher got the same unjam. |
| Final | Never dropped a bot in any run. Cost: slower (browser long bridge ~200 steps vs ~125; Python arch 97–151 vs 81–91). |

**Arm model + strength limit** (was stalling at 83–99% on every shape).

| Tried | Result |
| --- | --- |
| Hanging bots may only hang off fixed bots | Much worse: 0/8, 58–92%. Reverted. |
| Jam breaker: when stalled, hanging bots spend spells heading for the floor | Table and spiral finish. |
| **+ swivel: free bots turn their arms in place to link to more neighbours (keeping their holder and held bots)** | **7/8 shapes finish every time**: bridge 270, dome 215, pyramid 198, spiral 341, K 329, table 263, short 157. **The arch still jams in 7 of 8 runs** (95–99.7%): a dense pinned cluster on top. |

**Final browser runs** (local rule + holder tree + retrained model):

| Mode | Result |
| --- | --- |
| Cubes, rules | 11/11 shapes (bridge 197, arch 245, dome 173, spiral 266, ...), truss 3/3 |
| Cubes, strength limit | 8/8 (bridge 174, arch 377, dome 179, ...) |
| Arms, rules | 11/11 (bridge 241, arch 301, dome 321, ...) |
| Arms, strength limit | 7/8, arch jams (see above) |
| Trained model, cubes | wall 116, small arch 241, tower 92, short 116, table 166, pyramid 165, long bridge 184, dome 212 |
| Trained model, Real render | wall 107, small arch 267, tower 127, short 127, table 212 |

**Broken bots.** Click a bot with "Breaks its grip": it falls with 6% chance per step while off the floor. Each bot counts its own falls and retires to a reject pad after 2. Neighbours count failures per spot. Tests break 3 deck bots on a finished long bridge: usually 2–3 caught; one that falls only once and then parks on the floor is never used again, so it is never caught (harmless, but it's still out there).

**Arch "Cut the middle".** The cut removed the whole crown (~182 bots, with only 90 spares), so the repair could never finish (stuck at ~88%, also before today). Narrowed to a 2-cell slice (~56 bots): repaired in 85–93 steps with cubes, 105–124 with arm links.

**Training log.** `dagger.py` now writes `policy_log.json`; `plot_training.py` draws `media/linking_training.png`. Retrained on the local rules: Python wall/arch/tower 15/15 (arch 96–116 steps, rules 97–151).

## Still open

1. **Moves aren't real v1 moves.** Bots slide and pivot (or roll, on the triangle grid). v1 moves with two 3-joint arms. Needed: how does v1 actually get from one spot to the next?
2. **Arch with arm links + strength limit** jams most runs (section 8).
3. **Still global:** the hand-written rules' crawl map (the trained model uses a relayed signal and compass instead) and the support-pillar planner.
4. **Numbers to confirm:** 3 N per connector, whether tip-to-tip docking is possible, and at what spacing.
