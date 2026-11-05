"""A batched lattice world in a vertical plane (side view), on the GPU, with gravity and a cantilever limit.

Cells (row i = height above the floor, column j). The floor is under row 0. Bots sit in cells.
Moves (as in linking/: the bot keeps touching the structure while it moves):
  stay;
  slide E/W/U/D: one cell along a surface: both the origin and the destination touch the same side
                 (for E/W: both have something solid below, or both something above; for U/D: both have
                 something on the left, or both on the right);
  pivot (4 diagonals): roll over an edge: of the two corner cells, exactly one is solid (the pivot) and the
                 other is empty (the swept space).
All bots move at once; destinations must be empty at the start of the step; ties go to a random priority.
Physics (this replaces linking/'s "nothing may fall" rule and its support planner -- nothing is enforced):
  support span s: 0 for a bot standing on the floor; inherited from the bot below; 1 + s of a side or upper
  neighbour (a cantilever or a hanging bot). A bot is HELD if s <= L (the cantilever limit; the v1 design note says
  one bot can stick out, i.e. L = 1). Unheld bots cannot act and fall one cell per step until they land.
"""
import torch
import torch.nn.functional as F

# move index: 0 stay, 1 E, 2 W, 3 U, 4 D, 5 UE, 6 UW, 7 DE, 8 DW  (di = row change, dj = column change)
MOVES = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
INF = 1000.0


def shift(x, di, dj, fill=0.0):
    """out[..., i, j] = x[..., i - di, j - dj]  (content moves by (+di, +dj)); outside filled with `fill`."""
    H, W = x.shape[-2:]
    out = torch.full_like(x, fill)
    src_i = slice(max(0, -di), H - max(0, di))
    dst_i = slice(max(0, di), H - max(0, -di))
    src_j = slice(max(0, -dj), W - max(0, dj))
    dst_j = slice(max(0, dj), W - max(0, -dj))
    out[..., dst_i, dst_j] = x[..., src_i, src_j]
    return out


def look(x, di, dj, fill=0.0):
    """Value at the neighbour (i + di, j + dj) of each cell."""
    return shift(x, -di, -dj, fill)


def solid_at(occ, di, dj):
    """Is the neighbour (i+di, j+dj) solid? The floor (row -1) is solid, outside the grid otherwise is not."""
    s = look(occ, di, dj, 0.0)
    if di == -1:
        s = s.clone()
        s[..., 0, :] = 1.0
    return s


def support_span(occ, L, s0=None, iters=8):
    """Support span per cell (INF where empty or unheld). Bellman-style update from a warm start s0 (the previous
    step's values), `iters` rounds per call: values can go up as well as down, so removing a support is felt within
    a few steps (a short physical delay). L is finite; use a large L (e.g. 64) for 'no cantilever limit'."""
    floor = torch.zeros_like(occ)
    floor[..., 0, :] = 1.0
    s = torch.where(occ > 0, torch.full_like(occ, INF), torch.full_like(occ, INF)) if s0 is None else s0
    for _ in range(iters):
        below = look(s, -1, 0, INF)
        side = torch.minimum(look(s, 0, 1, INF), look(s, 0, -1, INF)) + 1
        above = look(s, 1, 0, INF) + 1
        cand = torch.minimum(torch.minimum(below, side), above)
        cand = torch.where(floor > 0, torch.zeros_like(cand), cand)
        cand = torch.where(cand <= L, cand, torch.full_like(cand, INF))     # unheld bots pass no support on
        s = torch.where(occ > 0, cand, torch.full_like(cand, INF))
    return s


def valid_moves(occ, held):
    """(B, 9, H, W) mask of allowed moves for a bot in each cell (0 where there is no bot)."""
    v = [torch.ones_like(occ)]
    for (di, dj) in MOVES[1:]:
        dest_empty = 1.0 - look(occ, di, dj, 1.0)        # outside the grid counts as blocked
        if di == 0:            # horizontal slide: same surface below or above
            surf = (solid_at(occ, -1, 0) * solid_at(occ, -1, dj)) + (solid_at(occ, 1, 0) * solid_at(occ, 1, dj))
            ok = dest_empty * (surf > 0).float()
        elif dj == 0:          # vertical slide: same wall left or right
            surf = (solid_at(occ, 0, -1) * solid_at(occ, di, -1)) + (solid_at(occ, 0, 1) * solid_at(occ, di, 1))
            ok = dest_empty * (surf > 0).float()
        else:                  # pivot over an edge
            a = solid_at(occ, di, 0)
            b = solid_at(occ, 0, dj)
            ok = dest_empty * ((a + b) == 1).float()
            if di == -1:       # cannot go below the floor
                ok = ok.clone(); ok[..., 0, :] = 0
        v.append(ok)
    v = torch.stack(v, 1)
    return v * (occ * held)[:, None]


def apply_moves(occ, hid, act, prio):
    """act: (B, H, W) long in 0..8 (only meaningful where occ). Resolves conflicts by priority.
    Returns new occ, new hid (hidden state travels with the bot), and the number of bots that moved."""
    best = torch.full_like(occ, -1.0)
    claims = []
    for k, (di, dj) in enumerate(MOVES[1:], start=1):
        mover = ((act == k).float() * occ)
        c = shift(prio * mover - (1 - mover), di, dj, -1.0)    # claim priority at the destination
        claims.append((k, di, dj, mover, c))
        best = torch.maximum(best, c)
    new_occ = occ.clone()
    new_hid = hid.clone()
    moved = torch.zeros_like(occ)
    for k, di, dj, mover, c in claims:
        win_dest = ((c >= 0) & (c >= best)).float()            # destination cells won by this direction
        win_src = look(win_dest, di, dj, 0.0) * mover           # the movers that won
        new_occ = new_occ - win_src + win_dest
        new_hid = new_hid * (1 - win_src[:, None]) + shift(hid * win_src[:, None], di, dj) * win_dest[:, None]
        moved = moved + win_src
    return new_occ.clamp(0, 1), new_hid, moved


def gravity(occ, hid, L, s):
    """Unheld bots (span s > L) fall one cell if the cell below is free. Returns occ, hid, number unheld."""
    unheld = occ * (s > L).float()
    below_free = 1.0 - solid_at(occ, -1, 0)
    fall = unheld * below_free
    new_occ = occ - fall + shift(fall, -1, 0)
    new_hid = hid * (1 - fall[:, None]) + shift(hid * fall[:, None], -1, 0)
    return new_occ, new_hid, unheld.flatten(1).sum(1)
