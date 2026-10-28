"""The linking swarm as an environment a trained model can drive.

Same world and rules as linking/lattice.py (cube bots, slides and pivots, no
move may leave anything unattached). What's new:

* observe(): what ONE bot can see, as 262 numbers (textbook L2, Parts 3 and 7)
* 19 action slots: stay, the 6 face cells (slides), the 12 edge cells (pivots)
* work_signal(): a "how far to the work" value passed bot to bot, the
  Kilobot hop-count gradient, instead of the global distance map
* step(): one tick where a chooser picks each free bot's move. The chooser
  is either the hand-written rules (the teacher) or a trained network.

The "will anything fall?" check is still global (is_connected).
"""

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lattice as L  # noqa: E402

GRID = L.GRID
R = 2                        # the bot sees cells up to 2 away: a 5x5x5 block
SIDE = 2 * R + 1
OBS_SIZE = 2 * SIDE ** 3 + 7 + 1 + 4  # 125 occupied + 125 shape + 7 signal + 1 locked + 4 compass = 262

# Action slots. Slot 0 = stay. Slots 1-6 = face cells. Slots 7-18 = edge cells.
FACES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
EDGES = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
         if abs(a) + abs(b) + abs(c) == 2]
OFFSETS = [(0, 0, 0)] + FACES + EDGES
N_ACTIONS = len(OFFSETS)                       # 19
SLOT_OF = {o: k for k, o in enumerate(OFFSETS)}

SIDEWAYS_CHANCE = L.SIDEWAYS_CHANCE

# "radio": hop count relayed between bots up to RADIO cells apart, like the
#          Kilobots' infrared (about 3 body widths). Still local.
# "local": hop count relayed only between touching bots.
# "global": the teacher's whole-grid distance map (a diagnostic only, to see
# whether the local signal is what holds the network back).
SIGNAL = os.environ.get("LINK_SIGNAL", "radio")
RADIO = 2


# ---------------------------------------------------------------- work signal

def work_signal(occ, goal, reach=1):
    """Hop count from the work, passed from bot to bot.

    Goal cells are 0. A bot touching a goal cell (or a bot that is 0) reads
    one more than the smallest value among its neighbours, and so on. Only
    bots carry the signal: it never jumps through empty air. This is what the
    bots would compute by talking to their neighbours a few times; here it is
    just computed directly. Cells nobody can reach this way get a large number.
    """
    big = 10_000
    dist = np.full(GRID, big, dtype=np.int32)
    frontier = goal.copy()
    dist[frontier] = 0
    seen = frontier.copy()
    k = 0
    carriers = occ | goal
    if reach == 1:
        links = L.FACE_LINKS                       # touching bots only
    else:                                          # any bot within `reach` cells
        links = np.ones((2 * reach + 1,) * 3, dtype=bool)
    while frontier.any():
        k += 1
        grown = ndimage.binary_dilation(frontier, structure=links)
        frontier = grown & ~seen & carriers
        dist[frontier] = k
        seen |= frontier
    return dist


def signal_feature(d):
    """Turn a hop count into a number between 0 and 1 (1 = right at the work)."""
    return 1.0 / (1.0 + d)


# ---------------------------------------------------------------- observation

def padded(grid, fill):
    """The grid with R extra cells on every side, so a 5x5x5 block can always be cut out."""
    return np.pad(grid, R, constant_values=fill)


def compass(goal_cells, cell):
    """Direction to the nearest open part of the shape, from the map and the bot's own position.

    Like the Kilobots: each bot is given the shape map and knows where it is,
    so it can work this out alone. 3 numbers for the direction (unit length)
    and 1 for closeness, 1/(1 + distance). All zero if nothing is open.
    """
    if len(goal_cells) == 0:
        return np.zeros(4, dtype=np.float32)
    d = goal_cells - np.asarray(cell)
    n = np.abs(d).sum(axis=1)
    k = int(np.argmin(n))
    v = d[k].astype(np.float32)
    length = float(np.linalg.norm(v))
    unit = v / length if length > 0 else v
    return np.array([unit[0], unit[1], unit[2], 1.0 / (1.0 + n[k])], dtype=np.float32)


def observe(state, cell):
    """262 numbers: what the bot in `cell` can see (or work out from the map)."""
    x, y, z = cell
    occ_p, tgt_p, sig_p = state["occ_pad"], state["tgt_pad"], state["sig_pad"]
    block_occ = occ_p[x:x + SIDE, y:y + SIDE, z:z + SIDE].ravel()
    block_tgt = tgt_p[x:x + SIDE, y:y + SIDE, z:z + SIDE].ravel()
    sig = [sig_p[x + R, y + R, z + R]]
    for dx, dy, dz in FACES:
        sig.append(sig_p[x + R + dx, y + R + dy, z + R + dz])
    locked = [1.0 if state["locked"][cell] else 0.0]
    return np.concatenate([block_occ, block_tgt, np.array(sig), np.array(locked),
                           compass(state["goal_cells"], cell)]).astype(np.float32)


def frame_state(sw, goal, dist=None):
    """Everything observe() needs for one tick, computed once."""
    occ_pad = padded(sw.occ.astype(np.float32), 0.0)
    occ_pad[:, :, :R] = 1.0                       # the floor counts as solid
    tgt_pad = padded(sw.target.astype(np.float32), 0.0)
    if SIGNAL == "global" and dist is not None:
        hops = dist
    else:
        hops = work_signal(sw.occ, goal, RADIO if SIGNAL == "radio" else 1)
    sig = signal_feature(hops).astype(np.float32)
    return {"occ_pad": occ_pad, "tgt_pad": tgt_pad, "sig_pad": padded(sig, 0.0),
            "locked": sw.locked, "goal_cells": np.argwhere(goal)}


# ---------------------------------------------------------------- one tick

def legal_slots(sw, p, busy):
    """Moves bot p could make right now: {slot: (kind, cell, via)}.

    Same geometry as lattice.legal_moves, minus cells that are locked or
    already used by another bot this tick. Safety (nothing falls) is checked
    later, only for the move actually tried.
    """
    out = {}
    for kind, c, via in L.legal_moves(sw.occ, p):
        if sw.locked[c] or busy[c] or busy[via]:
            continue
        off = tuple(int(c[i] - p[i]) for i in range(3))
        out[SLOT_OF[off]] = (kind, c, via)
    return out


def rule_order(sw, p, slots, dist):
    """The teacher: the hand-written rules, as an order of slots to try."""
    here = dist[p]
    ranked = []
    for s, (kind, c, via) in slots.items():
        if dist[c] < here:
            ranked.append((dist[c] + sw.rng.random() * 0.1, s))
        elif dist[c] == here and sw.rng.random() < SIDEWAYS_CHANCE:
            ranked.append((dist[c] + 0.5, s))
    ranked.sort()
    return [s for _, s in ranked]


def step(sw, chooser, record=None):
    """One tick of building. `chooser(sw, p, slots, ctx)` returns slots to try, best first.

    If `record` is a list, every free bot's (observation, legal-slot mask,
    slot it ended up taking) is appended to it. Returns the number of moves made.
    """
    sw.last_kind[:] = 0
    sw.last_from = sw.pos.copy()
    if sw.done():
        return 0
    goal = sw.goal_cells()
    ctx = {"dist": L.distance_field(sw.occ, sw.locked, goal), "goal": goal}
    ctx["frame"] = frame_state(sw, goal, ctx["dist"])
    busy = sw.occ.copy()
    free = np.argwhere(sw.occ & ~sw.locked)
    sw.rng.shuffle(free)
    moves = 0
    for p in free:
        p = tuple(int(v) for v in p)
        if not sw.occ[p] or sw.locked[p]:
            continue
        slots = legal_slots(sw, p, busy)
        obs = observe(ctx["frame"], p) if record is not None or chooser is not rule_chooser else None
        ctx["obs"] = obs
        taken = 0
        for s in chooser(sw, p, slots, ctx):
            if s == 0:
                break                                  # chose to stay
            kind, c, via = slots[s]
            sw.occ[p] = False
            sw.occ[c] = True
            if L.is_connected(sw.occ):                 # nothing comes loose
                bot = sw.id_at[p]
                sw.id_at[p], sw.id_at[c] = -1, bot
                sw.pos[bot] = c
                busy[c] = busy[via] = True
                moves += 1
                taken = s
                if goal[c]:
                    sw.locked[c] = True
                    goal[c] = False
                break
            sw.occ[c] = False
            sw.occ[p] = True
        if record is not None:
            mask = np.zeros(N_ACTIONS, dtype=bool)
            mask[0] = True
            for s in slots:
                mask[s] = True
            record.append((obs, mask, taken))
    # a bot standing on a target cell at or below the layer being built is in place
    empty_target = sw.target & ~sw.occ
    layers = np.where(empty_target.any(axis=(0, 1)))[0]
    top = layers[0] if len(layers) else GRID[2] - 1
    settled = sw.occ & sw.target
    settled[:, :, top + 1:] = False
    sw.locked |= settled
    return moves


def rule_chooser(sw, p, slots, ctx):
    return rule_order(sw, p, slots, ctx["dist"])


def build(shape, chooser, seed=0, max_steps=400, record=None, extra=12):
    """Build one shape from the starting pile. Returns (finished, steps, swarm)."""
    sw = L.Swarm(L.SHAPES[shape](), seed=seed, extra=extra)
    for t in range(max_steps):
        if sw.done():
            return True, t, sw
        step(sw, chooser, record)
    return sw.done(), max_steps, sw
