"""Linked modules on a 3D lattice: bots that stay attached while they move.

Every bot is a cube that sits in one cell of a 3D grid. Two bots that share a
face are linked. A bot can only move in two ways, both of which keep it
touching the structure while it moves:

* slide: move one cell sideways along a surface of other bots
* pivot: roll half a turn over the edge of the bot underneath, from its top
  face down onto its side face (going round a corner)

A move is only allowed if the whole structure is still one connected piece
afterwards, so nothing ever floats free.

Bots fill a target shape from the bottom up: a target cell opens once the
cell under it is filled (the readiness rule, see ready_cells). A movable bot crawls over
the outside of the structure, following a distance field that leads towards
the empty target cells, and stops once it reaches a target cell. Target cells
stay filled; bots inside the target never move again. Once the shape is done,
spare bots climb back down and park on the floor, ready for repairs.

What is NOT local here: the distance field and the "is everything still
connected?" check are computed over the whole grid each tick (linking/learn/
has local versions of both). A real swarm
would need local versions of both (a hop-count gradient like Phase 2, and a
local test for whether a bot is safe to remove). They are kept global here
so the movement rules can be studied on their own.

Run from the repo root:
    python linking/animate.py --shape sheet
"""

import numpy as np
from scipy import ndimage

# The 6 face directions: +x, -x, +y, -y, +z, -z.
FACES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]])

# Face-only (6-neighbour) connectivity for scipy's labelling.
FACE_LINKS = ndimage.generate_binary_structure(3, 1)

# One move reaches a face neighbour (slide) or an edge-diagonal cell (pivot):
# the 18 cells that share a face or an edge.
ONE_MOVE = ndimage.generate_binary_structure(3, 2)

GRID = (24, 16, 14)            # grid size in cells (x, y, z)
KEEP_LEVEL = True              # readiness rule also waits for the neighbours below (see ready_cells)
SIDEWAYS_CHANCE = 0.3          # chance of taking a move that doesn't get closer


# ---------------------------------------------------------------- shapes

def _empty():
    return np.zeros(GRID, dtype=bool)


def shape_sheet():
    """A thin vertical wall, one bot thick: the 'surface' test."""
    t = _empty()
    t[12:21, 7, 0:9] = True
    return t


def shape_arch():
    """Two legs and a bridge on top. Only holds together because bots link."""
    t = _empty()
    t[11:13, 6:9, 0:7] = True      # left leg
    t[19:21, 6:9, 0:7] = True      # right leg
    t[11:21, 6:9, 7:9] = True      # bridge
    return t


def shape_tower():
    """A hollow square tower: four thin walls."""
    t = _empty()
    t[13:19, 4:10, 0:8] = True
    t[14:18, 5:9, 0:8] = False
    return t


SHAPES = {"sheet": shape_sheet, "arch": shape_arch, "tower": shape_tower}


def initial_blocks(target, extra=12):
    """A compact pile of bots to the left of the target, touching the ground.

    The pile holds the number of target cells plus a few spares, laid out
    layer by layer so it starts as one connected piece.
    """
    n = int(target.sum()) + extra
    occ = _empty()
    count = 0
    for z in range(GRID[2]):
        for x in range(2, 10):
            for y in range(3, 12):
                if count == n:
                    return occ
                occ[x, y, z] = True
                count += 1
    return occ


# ---------------------------------------------------------------- checks

def in_grid(c):
    return all(0 <= c[i] < GRID[i] for i in range(3))


def solid(occ, c):
    """Is there something solid at cell c? The floor (z = -1) always is."""
    if c[2] == -1 and 0 <= c[0] < GRID[0] and 0 <= c[1] < GRID[1]:
        return True
    return in_grid(c) and bool(occ[tuple(c)])


def with_floor(occ):
    """The grid with an extra solid layer underneath, standing for the floor."""
    floor = np.ones(GRID[:2] + (1,), dtype=bool)
    return np.concatenate([floor, occ], axis=2)


def pieces(occ):
    """Label every face-linked piece. The floor counts as one big bot that
    never moves, so two towers standing on the floor are one structure."""
    labels, n = ndimage.label(with_floor(occ), structure=FACE_LINKS)
    return labels[:, :, 1:], n


def is_connected(occ):
    """True if all bots are linked to each other or to the floor."""
    _, n = pieces(occ)
    return n <= 1


def count_links(occ):
    """Number of shared faces between bots (each counted once)."""
    return int((occ[1:] & occ[:-1]).sum() + (occ[:, 1:] & occ[:, :-1]).sum()
               + (occ[:, :, 1:] & occ[:, :, :-1]).sum())


# ---------------------------------------------------------------- moves

def legal_moves(occ, p):
    """All cells bot p could reach in one slide or one pivot.

    Returns a list of (kind, end cell, via cell). `via` is the cell the bot
    passes through on the way, which must be empty: for a slide it is the
    end cell itself; for a pivot it is the corner cell the bot swings through
    halfway round.

    Both moves need something solid to push against, so a bot can never step
    off into empty space. The connectivity check comes later (it is the
    expensive part).
    """
    p = np.asarray(p)
    moves = []
    for d in FACES:
        step = p + d
        if not in_grid(step) or occ[tuple(step)]:
            continue
        for e in FACES:
            if abs(e @ d) != 0:          # e must be at right angles to d
                continue
            under_here = p + e
            under_there = step + e
            if not solid(occ, under_here):
                continue
            if solid(occ, under_there):
                # slide: a surface runs under both cells
                moves.append(("slide", tuple(step), tuple(step)))
            elif in_grid(under_there):
                # pivot: roll over the edge of the bot underneath, ending
                # on its side face at step + e, swinging through `step`.
                moves.append(("pivot", tuple(under_there), tuple(step)))
    # The same cell may be reachable more than one way; keep each once.
    seen, unique = set(), []
    for kind, c, via in moves:
        if c not in seen:
            seen.add(c)
            unique.append((kind, c, via))
    return unique


def distance_field(occ, locked, goal):
    """Moves from every cell to the nearest goal cell, crawling only.

    A path may only use cells that touch something fixed (a locked bot or
    the floor), because a bot can only move along a surface. Paths may pass
    through free bots (they will get out of the way) but not through locked
    ones. Cells with no crawling path get a large number.

    Paths only over free bots are added last: they can lead a bot down onto
    a fixed surface, but never up into the air (no building ladders).
    """
    big = 10_000
    dist = np.full(GRID, big, dtype=np.int32)
    fixed = with_floor(locked)
    touching = ndimage.binary_dilation(fixed, structure=FACE_LINKS)[:, :, 1:]
    passable = ~locked & touching
    frontier = goal.copy()
    dist[frontier] = 0
    reached = frontier.copy()
    k = 0
    while frontier.any():
        k += 1
        grown = ndimage.binary_dilation(frontier, structure=ONE_MOVE)
        frontier = grown & ~reached & passable
        dist[frontier] = k
        reached |= frontier

    # Bots resting on other free bots (high up in the pile, say) are not
    # touching anything fixed. Give them paths too, over free bots, that
    # lead down onto the fixed surfaces above.
    on_bots = ndimage.binary_dilation(with_floor(occ), structure=FACE_LINKS)[:, :, 1:]
    extra = ~locked & on_bots & ~reached
    longest = int(dist[reached].max()) if reached.any() else 0
    for k in range(1, longest + 60):
        grown = ndimage.binary_dilation(dist == k - 1, structure=ONE_MOVE)
        new = grown & extra & ~reached
        dist[new] = k
        reached |= new
    return dist


# Horizontal neighbours only (the 4 cells around a cell in its own layer).
FLAT_LINKS = np.zeros((3, 3, 3), dtype=bool)
FLAT_LINKS[1, 1, 1] = FLAT_LINKS[0, 1, 1] = FLAT_LINKS[2, 1, 1] = FLAT_LINKS[1, 0, 1] = FLAT_LINKS[1, 2, 1] = True

LEVEL_WINDOW = np.zeros((7, 7, 1), dtype=bool)
LEVEL_WINDOW[:, :, 0] = True                 # 7x7 in one layer: 3 cells around, about Kilobot infrared range


def ready_cells(target, locked):
    """Target cells that are ready to be filled, decided cell by cell.

    The readiness rule: a target cell is ready if
    * it is on the floor, or
    * the cell under it is part of the shape and already locked in, or
    * the cell under it is NOT part of the shape (an overhang, like the deck
      of a bridge) and a locked bot sits right beside it in the same layer;
    * and (KEEP_LEVEL) every shape cell in the layer below within 3 cells is
      filled, so no part of the shape runs far ahead of the rest.

    Every test looks only at a cell's own neighbours, so a real bot (or the
    cell's neighbours) could check it without knowing anything about the rest
    of the structure. It still builds bottom-up within each column.
    """
    below_locked = np.zeros_like(locked)
    below_locked[:, :, 1:] = locked[:, :, :-1]
    below_target = np.zeros_like(target)
    below_target[:, :, 1:] = target[:, :, :-1]
    beside_locked = ndimage.binary_dilation(locked, structure=FLAT_LINKS) & ~locked
    ready = np.where(below_target, below_locked, beside_locked)
    if KEEP_LEVEL:
        # don't run ahead: wait until the shape cells in the layer below, up to
        # 3 cells around, are filled too
        waiting = ndimage.binary_dilation(target & ~locked, structure=LEVEL_WINDOW)
        ready[:, :, 1:] &= ~waiting[:, :, :-1]
    ready[:, :, 0] = True
    return target & ready


class Swarm:
    """Holds the bots and the target, and runs one tick at a time."""

    def __init__(self, target, seed=0, extra=12):
        self.rng = np.random.default_rng(seed)
        self.target = target
        self.occ = initial_blocks(target, extra)
        self.locked = self.occ & target          # already in place
        self.moves_made = 0
        self.last_moved = []
        # Identity of each bot, so the animation can follow it smoothly.
        self.pos = np.argwhere(self.occ)         # (n_bots, 3) cell of each bot
        self.alive = np.ones(len(self.pos), dtype=bool)
        self.id_at = np.full(GRID, -1)           # which bot is in each cell
        self.id_at[tuple(self.pos.T)] = np.arange(len(self.pos))
        # What each bot did on the last tick: (kind, from, to). kind 0 = stay.
        self.last_kind = np.zeros(len(self.pos), dtype=int)
        self.last_from = self.pos.copy()
        self.last_via = self.pos.copy()          # corner cell a pivot swings through

    # -- which target cells are open for filling right now
    def goal_cells(self):
        """Empty target cells that are ready to fill (see ready_cells).

        Each cell decides this by looking only at its own neighbours, so
        there is no global "which layer are we on". Towers can grow at
        different speeds, and a hole always stays open from above, because the
        cell over it isn't ready until the hole is filled.
        """
        goal = ready_cells(self.target, self.locked) & ~self.occ
        if goal.any() or self.done():
            return goal
        # Shouldn't happen for a shape that stands on the floor; just in case,
        # open every empty target cell that touches something.
        touching = ndimage.binary_dilation(self.occ, structure=FACE_LINKS)
        touching[:, :, 0] = True
        return self.target & ~self.occ & touching

    def settle(self):
        """Lock any free bot already sitting on a target cell that is ready.

        Bottom-up, so a column of free bots standing in the shape locks in one go.
        """
        for z in range(GRID[2]):
            here = self.occ[:, :, z] & self.target[:, :, z] & ~self.locked[:, :, z]
            if not here.any():
                continue
            ready = ready_cells(self.target, self.locked)[:, :, z]
            self.locked[:, :, z] |= here & ready

    def damage(self, region):
        """Remove every bot inside `region` (a boolean grid). They're lost."""
        removed = self.occ & region
        self.occ &= ~region
        self.locked &= ~region
        self._drop(removed)
        # Any piece no longer linked to the floor falls away.
        labels, n = pieces(self.occ)
        if n > 1:
            floor_label = 1      # the floor layer is labelled first
            keep = labels == floor_label
            cut_off = self.occ & ~keep
            self._drop(cut_off)
            removed |= cut_off
            self.occ &= keep
            self.locked &= keep
        return int(removed.sum())

    def _drop(self, cells):
        ids = self.id_at[cells]
        self.alive[ids[ids >= 0]] = False
        self.id_at[cells] = -1

    def parking_cells(self):
        """Floor cells by the starting pile, where spare bots wait."""
        park = np.zeros(GRID, dtype=bool)
        park[2:10, 3:12, 0] = True
        return park & ~self.target

    def parked(self):
        """True once every spare bot is sitting in the parking area."""
        free = self.occ & ~self.locked
        return bool((free <= self.parking_cells()).all())

    def done(self):
        return bool((self.target <= self.occ).all())

    def tick(self):
        """Give each free bot, in random order, one chance to move."""
        self.last_moved = []
        self.last_kind[:] = 0
        self.last_from = self.pos.copy()
        self.last_via = self.pos.copy()
        building = not self.done()
        if building:
            goal = self.goal_cells()
        elif self.parked():
            return
        else:
            goal = self.parking_cells() & ~self.occ   # head back down
        if not goal.any():
            return

        dist = distance_field(self.occ, self.locked, goal)

        # All bots move at the same time, so a bot may only move through cells
        # that were empty when this step began and that no other bot has
        # claimed this step. (Otherwise two bots could pass through each other.)
        busy = self.occ.copy()

        free = np.argwhere(self.occ & ~self.locked)
        self.rng.shuffle(free)
        for p in free:
            p = tuple(p)
            if not self.occ[p] or self.locked[p]:
                continue
            here = dist[p]
            options = []
            for kind, c, via in legal_moves(self.occ, p):
                if self.locked[c] or busy[c] or busy[via]:
                    continue
                if dist[c] < here:
                    options.append((dist[c], kind, c, via))
                elif dist[c] == here and self.rng.random() < SIDEWAYS_CHANCE:
                    options.append((dist[c] + 0.5, kind, c, via))
            if not options:
                continue
            options.sort(key=lambda o: (o[0], self.rng.random()))
            for _, kind, c, via in options:
                # Try the move and keep it only if nothing comes loose.
                self.occ[p] = False
                self.occ[c] = True
                if is_connected(self.occ):
                    self.moves_made += 1
                    self.last_moved.append(c)
                    bot = self.id_at[p]
                    self.id_at[p], self.id_at[c] = -1, bot
                    self.pos[bot] = c
                    self.last_kind[bot] = 1 if kind == "slide" else 2
                    self.last_via[bot] = via
                    busy[c] = busy[via] = True
                    if building and goal[c]:
                        self.locked[c] = True
                        goal[c] = False
                    break
                self.occ[c] = False
                self.occ[p] = True

        if not building:
            return
        self.settle()


def run(shape="sheet", max_ticks=600, seed=0, damage_region=None, extra=12,
        pause=4, max_park=80):
    """Run until the shape is done (or max_ticks). Returns per-tick frames.

    If `damage_region` is given, the spare bots first climb down and park on
    the floor; `pause` ticks later every bot in the region is removed and the
    swarm rebuilds the shape from its spares.
    """
    target = SHAPES[shape]()
    swarm = Swarm(target, seed=seed, extra=extra)
    frames = []
    lost = 0
    damaged_at = None
    finished_at = None   # tick the shape was first complete
    settled_at = None    # tick all spares were parked
    for t in range(max_ticks):
        if swarm.done() and finished_at is None:
            finished_at = t
        if swarm.done() and swarm.parked() and settled_at is None:
            settled_at = t
        time_to_cut = False
        if damage_region is not None and damaged_at is None:
            if settled_at is not None and t == settled_at + pause:
                time_to_cut = True       # spares parked, short pause, cut
            if finished_at is not None and t == finished_at + max_park:
                time_to_cut = True       # parking took too long, cut anyway
        if time_to_cut:
            lost = swarm.damage(damage_region)
            damaged_at = t
        frames.append({
            "pos": swarm.pos.copy(),
            "from": swarm.last_from.copy(),
            "via": swarm.last_via.copy(),
            "kind": swarm.last_kind.copy(),
            "alive": swarm.alive.copy(),
            "bot_locked": swarm.locked[tuple(swarm.pos.T)] & swarm.alive,
            "occ": swarm.occ.copy(),
            "connected": is_connected(swarm.occ),
            "filled": float((swarm.occ & target).sum() / target.sum()),
            "damaged": damaged_at == t,
            "lost": lost,
        })
        repaired = damaged_at is not None and swarm.done()
        if (damage_region is None and swarm.done()) or repaired:
            break
        swarm.tick()                     # builds, parks, or holds still
    return target, frames, swarm


def cut_top(shape, z_from):
    """A damage region: every target cell at height z_from and above."""
    region = np.zeros(GRID, dtype=bool)
    region[:, :, z_from:] = True
    return region & SHAPES[shape]()


def cut_box(x0, x1, y0, y1, z0, z1):
    """A damage region: a box of cells (end values not included)."""
    region = np.zeros(GRID, dtype=bool)
    region[x0:x1, y0:y1, z0:z1] = True
    return region
