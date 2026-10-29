"""The linking swarm as an environment a trained model can drive.

Same world and rules as linking/lattice.py (cube bots, slides and pivots, no
move may leave anything unattached). What's new:

* observe(): what ONE bot can see, as 262 numbers (textbook L2, Parts 3 and 7)
* 19 action slots: stay, the 6 face cells (slides), the 12 edge cells (pivots)
* work_signal(): a "how far to the work" value passed bot to bot, the
  Kilobot hop-count gradient, instead of the global distance map
* step(): one tick where a chooser picks each free bot's move. The chooser
  is either the hand-written rules (the teacher) or a trained network.

* the "will anything fall?" check is local too (SAFETY = "tree"): every bot
  keeps a "holder", one linked neighbour (or the floor) that holds it up,
  and a bot may only move if each bot it holds can switch to another holder
  whose chain still reaches the floor without it.
  See HolderTree. SAFETY = "global" uses the old whole-structure check.
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

# "tree":   the local holder tree (HolderTree below)
# "global": flood the whole structure after every move (the old check)
SAFETY = os.environ.get("LINK_SAFETY", "tree")


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



# ---------------------------------------------------------------- local safety

FLOOR = -2          # holder value meaning "held by the floor"
EMPTY = -1
REACH = 16          # how many holders up the chain a bot will ask before giving up ("not sure")


class HolderTree:
    """The local "will anything fall?" check (the browser sandbox has the same one).

    Every bot keeps a holder: one linked neighbour that holds it up, or the
    floor. Bots locked into the shape hang only from other locked bots, so
    once a chain of holders reaches a locked bot (or the floor) it is anchored
    for good. Following holders never loops, so every chain ends at an anchor.

    * A free bot may move only if every bot it holds can switch to another
      linked neighbour whose own chain reaches an anchor within REACH hops
      without passing through the bot that's leaving. Finding out is a message
      passed up the chain, one neighbour at a time. The bots it holds are
      handed over one at a time, so one can hang from another that has
      already been moved off it.
    * After moving, a bot holds nobody, so it can take any linked neighbour
      as its holder (a locked one if it can).
    * Free bots switch to a neighbour with a shorter chain whenever there is
      one (relax()), so chains stay short and more bots are free to move.

    It is more cautious than checking the whole structure (a chain longer
    than REACH counts as "not sure"), but it never lets anything drop.
    """

    def __init__(self, occ, locked):
        self.holder = {}
        self.fixed = set()             # locked bots whose chain is all locked bots: anchors
        self.rebuild(occ, locked)

    def anchored(self, c):
        return self.holder.get(c) == FLOOR or c in self.fixed

    def chain_length(self, c, avoid=()):
        """Hops from c up its holders to an anchor, avoiding `avoid`; inf if not shown within REACH."""
        for k in range(REACH + 1):
            if c in avoid or c == EMPTY or c is None:
                return float("inf")
            if self.anchored(c):
                return k
            c = self.holder.get(c)
        return float("inf")

    def rebuild(self, occ, locked):
        """Start from scratch: a wave up from the floor, locked bots first (like the Kilobot gradient)."""
        self.holder, self.fixed = {}, set()
        seeds = [tuple(c) for c in np.argwhere(locked & occ) if c[2] == 0]
        for c in seeds:
            self.holder[c] = FLOOR
        for c in self._wave(locked & occ, seeds):
            self.fixed.add(c)
        floor_free = [tuple(c) for c in np.argwhere(occ & ~locked) if c[2] == 0]
        for c in floor_free:
            self.holder[c] = FLOOR
        self._wave(occ, floor_free + list(self.fixed))

    def _wave(self, members, seeds):
        seeds = list(seeds)
        for c in seeds:
            for n in neighbours(c):
                if members[n] and n not in self.holder:
                    self.holder[n] = c
                    seeds.append(n)
        return seeds

    def plan_leave(self, occ, p):
        """If p can leave, [(bot it holds, new holder)]; otherwise None."""
        todo = [n for n in neighbours(p) if occ[n] and self.holder.get(n) == p]
        old = {n: self.holder[n] for n in todo}
        plan = []
        while todo:
            left = []
            for n in todo:
                if n[2] == 0:
                    plan.append((n, FLOOR)); self.holder[n] = FLOOR
                    continue
                best, best_len = None, float("inf")
                for q in neighbours(n):
                    if q != p and occ[q]:
                        length = self.chain_length(q, (p, n))
                        if length < best_len:
                            best, best_len = q, length
                if best is None:
                    left.append(n)
                else:
                    plan.append((n, best)); self.holder[n] = best   # tentatively
            if len(left) == len(todo):
                break
            todo = left
        self.holder.update(old)                                     # put back; apply() makes it real
        return None if len(plan) < len(old) else plan

    def attach(self, occ, locked, c):
        """Give the bot at c (holding nobody) a holder: an anchored locked one if it can."""
        if c[2] == 0:
            self.holder[c] = FLOOR
            if locked[c]:
                self.fixed.add(c)
            return
        nbrs = [q for q in neighbours(c) if occ[q]]
        best = min(nbrs, key=lambda q: (q not in self.fixed, min(self.chain_length(q, (c,)), REACH + 1)))
        self.holder[c] = best
        if locked[c] and best in self.fixed:
            self.fixed.add(c)

    def apply(self, occ, locked, p, c, plan):
        """p has moved to c (occ and locked already show it). Hand over p's bots, then attach c."""
        for n, q in plan:
            self.holder[n] = q
        self.holder.pop(p, None)
        self.fixed.discard(p)
        self.attach(occ, locked, c)

    def lock(self, occ, locked, cells):
        """Newly locked bots move onto an anchored locked neighbour, lowest first."""
        for c in sorted(cells, key=lambda c: c[2]):
            self.settle_on_locked(occ, c)

    def settle_on_locked(self, occ, c):
        if c[2] == 0:
            self.holder[c] = FLOOR
            self.fixed.add(c)
            return
        for q in neighbours(c):
            if occ[q] and q in self.fixed:
                self.holder[c] = q
                self.fixed.add(c)
                return

    def relax(self, occ, locked):
        """Free bots switch to a neighbour with a shorter chain (one not running through themselves)."""
        for c in map(tuple, np.argwhere(occ)):
            if locked[c]:
                if c not in self.fixed:
                    self.settle_on_locked(occ, c)
                continue
            if c[2] == 0:
                self.holder[c] = FLOOR
                continue
            best, best_len = self.holder.get(c), self.chain_length(c) - 1
            for q in neighbours(c):
                if occ[q]:
                    length = self.chain_length(q, (c,))
                    if length < best_len:
                        best, best_len = q, length
            self.holder[c] = best


def neighbours(c):
    x, y, z = c
    for dx, dy, dz in FACES:
        n = (x + dx, y + dy, z + dz)
        if 0 <= n[0] < GRID[0] and 0 <= n[1] < GRID[1] and 0 <= n[2] < GRID[2]:
            yield n


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
        elif getattr(sw, "stalls", 0) > 3 and sw.rng.random() < 0.3:
            # unjam: after a few steps where nothing new got locked in, a bot
            # may step backwards too, so a knot of bots can come apart
            ranked.append((10**6 + sw.rng.random(), s))
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
    if SAFETY == "tree":
        if not hasattr(sw, "tree"):
            sw.tree = HolderTree(sw.occ, sw.locked)
        sw.tree.relax(sw.occ, sw.locked)
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
        plan = sw.tree.plan_leave(sw.occ, p) if SAFETY == "tree" else []
        for s in chooser(sw, p, slots, ctx):
            if s == 0 or plan is None:
                break                                  # chose to stay, or it's holding someone up
            kind, c, via = slots[s]
            sw.occ[p] = False
            sw.occ[c] = True
            ok = True if SAFETY == "tree" else L.is_connected(sw.occ)   # nothing comes loose
            if ok:
                bot = sw.id_at[p]
                sw.id_at[p], sw.id_at[c] = -1, bot
                sw.pos[bot] = c
                busy[c] = busy[via] = True
                moves += 1
                taken = s
                if goal[c]:
                    sw.locked[c] = True
                    goal[c] = False
                if SAFETY == "tree":
                    sw.tree.apply(sw.occ, sw.locked, p, c, plan)
                break
            sw.occ[c] = False
            sw.occ[p] = True
        if record is not None:
            mask = np.zeros(N_ACTIONS, dtype=bool)
            mask[0] = True
            for s in slots:
                mask[s] = True
            record.append((obs, mask, taken))
    before = sw.locked.copy()
    sw.settle()      # free bots standing on ready target cells lock in place
    progress = int(sw.locked.sum()) > getattr(sw, "locked_before", 0)
    sw.locked_before = int(sw.locked.sum())
    sw.stalls = 0 if progress else getattr(sw, "stalls", 0) + 1
    if SAFETY == "tree":
        sw.tree.lock(sw.occ, sw.locked, list(map(tuple, np.argwhere(sw.locked & ~before))))
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
