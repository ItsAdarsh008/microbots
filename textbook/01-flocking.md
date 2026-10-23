# Chapter 1: Flocking with local rules

**Phase 1 · Code:** `phase1/flock.py`, `phase1/animate.py`, `phase1/checks.py`

![1,000 bots forming flocks, then gathering in a target zone](../media/flock.gif)

In Chapter 0 the bots ignored each other. Now each bot looks at the bots near it and reacts, and something nobody programmed appears: **flocks**. Groups of bots line up and travel together, with no leader and no bot knowing what the whole swarm is doing. Halfway through, a beacon switches on and the swarm gathers in a target zone.

This is the first real piece of swarm behaviour in the project, and it introduces the question every later phase depends on: **how does a bot find its neighbours quickly?**

This chapter assumes you've done Chapter 0: arrays and shapes, keys, `jit`, `vmap` and `scan`.

---

## Run it first

From the repo root, with `.venv` active:

```powershell
python phase1/flock.py      # prints alignment and how many bots reached the zone
python phase1/checks.py     # should end with 10/10 checks passed (takes ~30 s)
python phase1/animate.py    # opens the animation in a window
```

In the animation, **colour shows direction**: each dot is coloured by the way it's heading (red one way, green another, blue another, and so on round the colour wheel). At the start the colours are mixed, because every bot heads somewhere random. As flocks form, patches of the same colour appear, because the bots in each flock are heading the same way.

---

## Part 1: The idea

### 1.1 Emergence: simple local rules, complex group behaviour

In 1986 Craig Reynolds showed that you can get realistic bird flocks from three simple rules, each using only what a bird can see nearby. He called his simulated birds **boids**. No boid knows the flock's shape or has a plan; the flock **emerges** from thousands of small local decisions.

That's exactly the "no central brain" idea behind the microbots, which is why it's Phase 1.

### 1.2 What a bot can sense

Each bot only notices other bots closer than `VIEW_RADIUS` (5 units, in a world 100 units wide). Those are its **neighbours**. Anything further away doesn't exist as far as that bot is concerned.

```python
VIEW_RADIUS = 5.0     # neighbours closer than this are "seen"
SEP_RADIUS = 1.5      # seen neighbours closer than this are "too close"
WALL_MARGIN = 8.0     # walls are felt within this distance
```

### 1.3 The three rules

Every step, each bot works out three "wishes" from its neighbours:

| Rule | The wish | How it's computed |
| --- | --- | --- |
| **Separation** | don't bump into the bots right next to me | push away from every neighbour closer than `SEP_RADIUS`, harder the closer it is |
| **Alignment** | go the way my neighbours are going | the difference between my velocity and their average velocity |
| **Cohesion** | stay with the group | the direction to the average position of my neighbours |

Each wish is a small vector (a nudge to the velocity). They're scaled by weights and added together:

```python
W_SEPARATION = 0.30
W_ALIGNMENT = 0.08
W_COHESION = 0.01
```

The weights are the personality of the swarm. Separation is strongest because collisions matter most; cohesion is weakest because a strong pull to the group's centre makes bots clump into tight balls. These values were tuned by trying them and watching.

### 1.4 Two forces from the environment

Two more nudges don't come from other bots:

- **Walls:** a bot within `WALL_MARGIN` of a wall feels a push away from it, stronger the closer it gets. This is like a bot sensing an obstacle; it's still local.
- **Beacon:** once switched on, every bot can sense the direction of the target zone, like a light it can see from anywhere. That's information from the environment, not from a leader or from the other bots, so the "no central brain" rule still holds.

---

## Part 2: The rules in code

Open `phase1/flock.py`, section 2.

### 2.1 `steer_one`: the logic for one bot

As in Chapter 0, the rules are written for **one bot** and `vmap` runs them for every bot. Here's the start:

```python
def steer_one(i, candidates, valid, pos, vel, beacon_on):
    p, v = pos[i], vel[i]
    other_p, other_v = pos[candidates], vel[candidates]
```

New in this chapter: `pos[candidates]`. When you index an array with **another array of indices**, you get those rows back in that order. If `candidates` is `[4, 17, 9]`, then `pos[candidates]` is the positions of bots 4, 17 and 9. This is called **gather** (or "fancy indexing"), and it's how a bot pulls in the data of the bots it might see.

`candidates` comes from a **neighbour finder** (Part 3). It's a list of bots that *might* be close, and `valid` marks which entries in that list are real.

### 2.2 Who does this bot actually see?

```python
    offset = other_p - p                                   # vectors to each candidate
    dist = jnp.linalg.norm(offset, axis=1)
    seen = valid & (candidates != i) & (dist < VIEW_RADIUS)
    too_close = seen & (dist < SEP_RADIUS)
    count = jnp.maximum(seen.sum(), 1)                     # avoid dividing by 0
    any_seen = seen.any()
```

- `offset` is an arrow from this bot to each candidate. `jnp.linalg.norm(..., axis=1)` gives each arrow's length: the distance, by Pythagoras (√(x² + y²)).
- `seen` is a **mask**: `True` for candidates that are real, aren't this bot itself, and are within view. `&` means "and", applied cell by cell.
- `count` is how many neighbours it sees. `jnp.maximum(..., 1)` stops a division by zero for a lonely bot with no neighbours.

**Masks instead of `if`:** the code never says "if this neighbour is close, add it". Instead it computes something for *every* candidate and multiplies by the mask, so entries that don't count become zero. You'll see `* seen[:, None]` below. This is the same idea as `jnp.where` in Chapter 0: it keeps the code jit-friendly, because the shapes never depend on how many neighbours a bot happens to have.

`seen[:, None]` turns a mask of shape `(M,)` into shape `(M, 1)`, so it can multiply an `(M, 2)` array row by row (both x and y of each row get the same 0 or 1). This automatic stretching of shapes is called **broadcasting**.

### 2.3 The three rules, line by line

```python
    # Separation: push away from bots that are too close, harder when closer.
    push = -offset / jnp.maximum(dist, 1e-3)[:, None] ** 2
    separation = (push * too_close[:, None]).sum(axis=0)
```

`-offset` points *away* from each neighbour. Dividing by the distance squared makes the push grow fast as bots get closer: half the distance, four times the push. `jnp.maximum(dist, 1e-3)` avoids dividing by zero if two bots land on the same spot. `.sum(axis=0)` adds up the pushes from all the too-close neighbours into one vector.

```python
    # Alignment: match the average velocity of the bots you can see.
    avg_vel = (other_v * seen[:, None]).sum(axis=0) / count
    alignment = jnp.where(any_seen, avg_vel - v, 0.0)
```

The average velocity of the seen neighbours, minus my own velocity: the change that would make me move like them. If I see nobody, `jnp.where` makes it zero (otherwise the "average" of nobody, 0, would tell me to stop).

```python
    # Cohesion: move toward the average position of the bots you can see.
    cohesion = (offset * seen[:, None]).sum(axis=0) / count
```

The average of the arrows to my neighbours is an arrow to the centre of my group.

### 2.4 Walls and beacon

```python
def wall_push(p):
    near_low = jnp.maximum(0.0, WALL_MARGIN - p)          # near left / bottom
    near_high = jnp.maximum(0.0, WALL_MARGIN - (BOX - p))  # near right / top
    return (near_low - near_high) / WALL_MARGIN
```

For a bot at x = 3, `WALL_MARGIN - p` is 8 − 3 = 5, so it's pushed right with strength 5/8. At x = 50 both terms are negative, `jnp.maximum(0.0, …)` turns them into 0, and there's no push. The same code handles x and y at once, because `p` holds both.

```python
    # Beacon: once switched on, head for the target zone.
    to_target = TARGET - p
    target_dist = jnp.linalg.norm(to_target)
    # Full strength outside the zone, fading toward the centre inside it,
    # so bots settle into the zone instead of flying straight through.
    strength = jnp.minimum(target_dist / TARGET_RADIUS, 1.0)
    direction = to_target / jnp.maximum(target_dist, 1e-6)
    beacon = jnp.where(beacon_on, strength * direction, 0.0)
```

`direction` is a **unit vector** (length 1) pointing at the target: the arrow divided by its own length. Outside the zone the pull is full strength; inside it fades the closer a bot gets to the centre.

An earlier version switched the pull off completely inside the zone. Bots then flew straight through and out the other side, because alignment kept them going, and only about half the swarm was in the zone at any time. The fading pull fixed that: bots settle in and swirl around the centre. Tuning like this is a normal part of building swarm behaviour.

### 2.5 Adding it up, and `vmap` with `in_axes`

```python
    return (W_SEPARATION * separation
            + W_ALIGNMENT * alignment
            + W_COHESION * cohesion
            + W_WALL * wall_push(p)
            + W_BEACON * beacon)


steer_all = jax.vmap(steer_one, in_axes=(0, 0, 0, None, None, None))
```

`in_axes` tells `vmap` how to treat each argument:

| Argument | `in_axes` | Meaning |
| --- | --- | --- |
| `i` | 0 | one bot index per run |
| `candidates` | 0 | one row of candidates per bot |
| `valid` | 0 | one row of flags per bot |
| `pos`, `vel` | `None` | the **whole** swarm, shared by every run |
| `beacon_on` | `None` | one value shared by everyone |

`None` is what lets every bot look up its neighbours in the full `pos` and `vel` arrays.

### 2.6 Moving: speed limits

```python
def move(pos, vel, steering):
    vel = vel + steering
    speed = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel / jnp.maximum(speed, 1e-6) * jnp.clip(speed, MIN_SPEED, MAX_SPEED)
    pos = pos + vel
```

After steering, each velocity is split into direction (`vel / speed`) and speed, the speed is clamped between `MIN_SPEED` and `MAX_SPEED`, and the two are put back together. The minimum stops bots from stalling; the maximum stops them from zooming off. `keepdims=True` keeps the speed's shape as `(N, 1)` instead of `(N,)`, so it broadcasts against `(N, 2)`. The function ends with the same hard-wall bounce as Chapter 0, as a safety net.

---

## Part 3: Finding neighbours fast

This is the most important computer-science idea in the chapter.

### 3.1 The problem with checking everyone

The simplest way to find neighbours: every bot measures its distance to **every** other bot.

```python
def brute_force_neighbours(pos):
    n = pos.shape[0]
    candidates = jnp.broadcast_to(jnp.arange(n), (n, n))
    valid = jnp.ones((n, n), dtype=bool)
    return candidates, valid
```

Every bot gets all N bots as candidates. That's N × N distances: 1 million for 1,000 bots, 25 million for 5,000. Doubling the swarm makes the work **four times** bigger. This is called **O(N²)** ("order N squared") growth, and it's why brute force gets slow fast.

But almost all of that work is wasted. A bot can only see 5 units, and nearly every other bot is further away than that.

### 3.2 The cell list: only check nearby cells

Cut the world into a grid of square **cells**, each `VIEW_RADIUS` wide (20 × 20 cells). Any bot within view of you must be in **your cell or one of the 8 cells around it**, because nothing in a cell further away can be closer than one cell width. So each bot only checks those 9 cells.

```python
GRID = math.ceil(BOX / VIEW_RADIUS)   # 20 x 20 cells, each VIEW_RADIUS wide
CELL_SIZE = BOX / GRID
MAX_PER_CELL = 32                     # how many bots we look at per cell
OFFSETS = jnp.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
```

`OFFSETS` lists the 9 steps from your cell to each cell in the 3 × 3 block: (−1, −1), (−1, 0), … (1, 1).

The work now grows with N instead of N², because each bot checks a roughly fixed number of nearby bots however big the swarm is.

### 3.3 How the cell list is built

```python
    cells = jnp.clip(jnp.floor(pos / CELL_SIZE).astype(jnp.int32), 0, GRID - 1)
    cell_id = cells[:, 0] * GRID + cells[:, 1]
```

**Which cell is each bot in?** Divide its position by the cell size and round down: a bot at (23.7, 8.1) is in cell column 4, row 1. `jnp.clip` keeps bots exactly on the far wall inside the grid. Then each (column, row) pair becomes one number, `column × 20 + row`, so cells are numbered 0 to 399.

```python
    order = jnp.argsort(cell_id)
    sorted_id = cell_id[order]
```

**Sort the bots by cell.** `jnp.argsort` doesn't return the sorted values; it returns the *order* to put the bots in. After sorting, all the bots in cell 0 come first, then all the bots in cell 1, and so on, so each cell's bots sit together in one run.

```python
    every_cell = jnp.arange(GRID * GRID)
    start = jnp.searchsorted(sorted_id, every_cell, side="left")
    end = jnp.searchsorted(sorted_id, every_cell, side="right")
```

**Find where each cell's run starts and ends.** `jnp.searchsorted` finds where a value would go in a sorted list. For cell 37, `start[37]` is where cell 37's bots begin in the sorted list and `end[37]` is just past the last one, so `end[37] - start[37]` is how many bots it holds. An empty cell has `start == end`.

```python
    def candidates_for(cell):
        nearby = cell + OFFSETS                               # (9, 2)
        on_grid = ((nearby >= 0) & (nearby < GRID)).all(axis=1)
        nearby = jnp.clip(nearby, 0, GRID - 1)
        nearby_id = nearby[:, 0] * GRID + nearby[:, 1]        # (9,)

        slot = start[nearby_id][:, None] + jnp.arange(MAX_PER_CELL)[None, :]
        valid = on_grid[:, None] & (slot < end[nearby_id][:, None])
        idx = order[jnp.clip(slot, 0, n - 1)]
        return idx.reshape(-1), valid.reshape(-1)

    return jax.vmap(candidates_for)(cells)
```

**Collect the candidates from the 9 cells.** For one bot's cell:

1. `nearby` is the 9 surrounding cells. Cells off the edge of the grid are marked invalid in `on_grid`.
2. For each of those cells, look at up to `MAX_PER_CELL` slots in the sorted list: `start`, `start + 1`, `start + 2`, … That gives a `(9, 32)` grid of slots.
3. A slot is `valid` only if it's before that cell's `end` (and the cell is on the grid).
4. `order[slot]` converts sorted positions back to real bot indices.
5. `.reshape(-1)` flattens the `(9, 32)` grid into one list of 288 candidates.

`vmap` runs this for every bot. The result has the same `(candidates, valid)` form as brute force, which is why `steer_one` works with either finder unchanged.

### 3.4 Why a fixed `MAX_PER_CELL`?

JAX needs fixed shapes (Chapter 0, part 2.8), but cells hold different numbers of bots. So every cell gets exactly 32 slots, and `valid` switches off the unused ones. The cost: if a cell ever holds **more** than 32 bots, the extras are silently missed. That's why `checks.py` measures the busiest cell over the whole run (25 at most here) and fails if it ever goes over the limit.

### 3.5 The speed-up

From `checks.py` on one test machine:

| Bots | Brute force | Cell list | Speed-up |
| --- | --- | --- | --- |
| 1,000 | 41 steps/s | 195 steps/s | about 5× |
| 5,000 | 2 steps/s | 26 steps/s | about 13× |

The gap widens as the swarm grows, exactly as O(N²) versus O(N) predicts. Your machine's numbers will differ, but the pattern will be the same.

---

## Part 4: Running and measuring the swarm

### 4.1 `simulate`

```python
@partial(jax.jit, static_argnames=("n_bots", "n_steps", "beacon_from", "use_cells"))
def simulate(key, n_bots=1000, n_steps=600, beacon_from=300, use_cells=True):
    pos, vel = initial_state(key, n_bots)

    def tick(carry, t):
        pos, vel = carry
        pos, vel = step(pos, vel, t >= beacon_from, use_cells)
        return (pos, vel), (pos, vel)

    _, (traj_pos, traj_vel) = jax.lax.scan(tick, (pos, vel), jnp.arange(n_steps))
    return traj_pos, traj_vel
```

The same `scan` pattern as Chapter 0, with two differences:

- `scan` is given `jnp.arange(n_steps)` as its sequence, so `tick` receives the step number `t` each time. That's how the beacon knows when to switch on: `t >= beacon_from`.
- Each step saves **two** things, positions and velocities, so `scan` returns two stacked arrays. The velocities are needed to colour the dots by heading.

There's no randomness after the starting positions: given the same start, the flock always moves the same way. The randomness in Chapter 0 came from noise; here the interesting motion comes from the bots reacting to each other.

### 4.2 Measuring "flockiness": local alignment

```python
def local_alignment(pos, vel):
    heading = vel / jnp.maximum(jnp.linalg.norm(vel, axis=1, keepdims=True), 1e-6)
    dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    seen = dist < VIEW_RADIUS                       # includes the bot itself
    mean_heading = (seen[:, :, None] * heading[None, :, :]).sum(1) / seen.sum(1)[:, None]
    return jnp.linalg.norm(mean_heading, axis=1).mean()
```

For each bot, average the **headings** (unit-length velocities) of the bots it can see, including itself, and measure the length of that average. If they all point the same way, the arrows add up to length 1. If they point every which way, they cancel out and the length is near 0. Averaging over all bots gives one number for the swarm: about 0.25 at the start, about 0.9 once flocks have formed.

`pos[:, None, :] - pos[None, :, :]` is broadcasting again: it makes an `(N, N, 2)` array of every bot's offset to every other bot. It's brute force, which is fine for a measurement that runs a handful of times.

---

## Part 5: What the checks prove

| Check | What a failure would mean |
| --- | --- |
| Shape is `(600, 1000, 2)` | `scan` is stacking the wrong thing |
| Every bot stays inside the world | walls are broken |
| Speeds stay within the limits | `move` isn't clamping correctly |
| No cell overflows `MAX_PER_CELL` | the cell list is silently missing neighbours |
| Cell list matches brute force | the cell list has a bug (wrong cells, wrong slots) |
| A far-away bot doesn't affect bot 0 | the rules secretly use information from outside the view radius |
| A nearby bot does affect bot 0 | the bot isn't really reacting to its neighbours |
| Bots line up with their neighbours | flocking isn't emerging (weights or rules are off) |
| Most bots end up in the target zone | the beacon isn't working |
| 1,000 bots at 30+ steps per second | too slow for a live demo |

The two **locality** checks are the proof behind "no central brain". They move one bot and test whether bot 0's steering changes: it must change when the moved bot is nearby, and must not change at all when it's far away. That's exactly what "each bot only uses local information" means.

---

## Try this

Predict first, then run `python phase1/animate.py`. Undo each change afterwards.

1. Set `W_ALIGNMENT = 0.0`. Do flocks still form? What do they look like?
2. Set `W_COHESION = 0.1` (ten times stronger). What happens to the flocks' shape?
3. Set `W_SEPARATION = 0.0`. Watch closely: what goes wrong? Then run `checks.py`. Which check fails, and why does that make sense?
4. Change `VIEW_RADIUS` to 2.5, then to 10. How does flock size change? What happens to `GRID`, and to the speed numbers in `checks.py`?
5. Set `MAX_PER_CELL = 8`. Which check fails? Look at its detail text, and explain in your own words what the cell list is now getting wrong.
6. Move `TARGET` to `[20.0, 80.0]` and make the beacon switch on at step 100 (`python phase1/animate.py --beacon-from 100`).
7. Harder: add a fourth rule that makes bots avoid a circular obstacle in the middle of the world, the same way `wall_push` avoids walls.

---

## Check your understanding

<details>
<summary>1. What does "emergent" mean here? Where in the code is the flock?</summary>

Nowhere. No line of code describes a flock. Each bot only follows three local rules about its neighbours; the flocks appear from many bots doing that at once. That's emergence.
</details>

<details>
<summary>2. What are the three boids rules, and why is cohesion's weight so small?</summary>

Separation (avoid crowding the closest bots), alignment (match neighbours' velocity) and cohesion (move toward neighbours' average position). Cohesion is small because a strong pull to the centre squeezes groups into tight balls; separation is largest because avoiding collisions matters most.
</details>

<details>
<summary>3. Why does the code multiply by masks like <code>seen[:, None]</code> instead of using an <code>if</code>?</summary>

So the shapes never depend on how many neighbours a bot has. Every candidate is computed, and the mask zeroes out the ones that don't count. That keeps the function compatible with `jit` and `vmap`, which need fixed shapes and can't branch on values.
</details>

<details>
<summary>4. What does <code>pos[candidates]</code> do?</summary>

It gathers rows: it returns the positions of the bots whose indices are listed in `candidates`, in that order.
</details>

<details>
<summary>5. Why is brute force O(N²), and why does the cell list grow only like N?</summary>

Brute force has every bot check every bot: N × N pairs, so doubling N quadruples the work. With a cell list, each bot only checks the bots in 9 nearby cells, roughly a fixed number however big the swarm gets, so total work is about N × a constant.
</details>

<details>
<summary>6. Why must cells be at least <code>VIEW_RADIUS</code> wide?</summary>

So that every bot within view is guaranteed to be in your cell or an adjacent one. If cells were smaller, a visible bot could be two cells away and the 3 × 3 search would miss it.
</details>

<details>
<summary>7. What goes wrong if a cell holds more than <code>MAX_PER_CELL</code> bots?</summary>

The extra bots beyond the 32 slots aren't looked at, so some real neighbours are silently ignored. The fixed size is needed because JAX requires fixed shapes; the overflow check catches it.
</details>

<details>
<summary>8. In <code>vmap(steer_one, in_axes=(0, 0, 0, None, None, None))</code>, why are <code>pos</code> and <code>vel</code> given <code>None</code>?</summary>

Every bot needs the whole swarm's arrays so it can look up its candidates' positions and velocities. `None` means "pass the same full array to every run" instead of splitting it one row per bot.
</details>

<details>
<summary>9. The beacon is sensed by every bot. Doesn't that break "no central brain"?</summary>

No. The beacon is part of the environment, like a light every bot can see. No bot receives orders from another bot, and no bot knows where the others are beyond its view radius. The locality checks confirm that each bot's steering depends only on nearby bots.
</details>

<details>
<summary>10. What does local alignment measure, and what values did the swarm reach?</summary>

The average, over all bots, of how much each bot's neighbours (including itself) point the same way: 0 for random headings, 1 for perfectly matched. It started around 0.25 and reached about 0.9 after 300 steps of free flocking.
</details>

---

## Glossary

| Term | Meaning |
| --- | --- |
| Boids | Reynolds' 1986 simulated birds, flocking from three local rules |
| Emergence | Group behaviour that no individual rule describes |
| Neighbour | Another bot within `VIEW_RADIUS` |
| Separation / alignment / cohesion | Avoid crowding / match heading / stay with the group |
| Unit vector | An arrow of length 1, used for direction only |
| Mask | An array of `True`/`False` used to switch entries on or off |
| Broadcasting | NumPy and JAX stretching a size-1 axis to match another array's shape |
| Gather | Indexing with an array of indices to pull out those rows |
| Neighbour finder | A function that lists which bots each bot might see |
| Brute force | Checking every pair; O(N²) |
| Cell list | Splitting space into cells so each bot only checks nearby cells; about O(N) |
| `argsort` | Returns the order that would sort an array |
| `searchsorted` | Finds where values would go in a sorted array |
| Local alignment | How much neighbours' headings agree, from 0 (random) to 1 (identical) |
| Locality | Each bot's behaviour depends only on what's within its sensing range |
