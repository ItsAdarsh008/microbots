# Chapter 2: Self-assembly into a shape

**Phase 2 · Code:** `phase2/assemble.py`, `phase2/animate.py`, `phase2/checks.py`

![150 robots assembling into a star](../media/assemble_star.gif)

In Chapter 1 the swarm moved together, but it couldn't *build* anything. Here, 150 robots start packed in a cluster and assemble themselves into a chosen shape (a star, a wall, or the letter K) one robot at a time. No robot is told where to go. Each one works out where it is, travels around the outside of the group, and decides for itself where to stop.

This is a simplified recreation of a famous experiment: **Rubenstein, Cornejo and Nagpal, "Programmable self-assembly in a thousand-robot swarm" (Science, 2014)**, where 1,024 real Kilobot robots did this over about 12 hours. Building the K is a nod to that paper, which built one too.

This chapter assumes Chapters 0 and 1. It introduces four new ideas: **gradients**, **trilateration**, **edge-following** and **state machines**.

---

## Run it first

From the repo root, with `.venv` active:

```powershell
python phase2/assemble.py --shape star   # prints how many joined and how well the shape is covered (~45 s)
python phase2/animate.py --shape k       # opens the animation (simulates first, then plays)
python phase2/checks.py                  # should end with 12/12 checks passed (~2-3 minutes)
```

Colours in the animation: **grey** = waiting, **orange** = moving outside the shape, **yellow** = moving inside the shape, **purple** = joined, **green** = seed. The faint grey area is the target shape.

---

## Part 1: The idea

### 1.1 What each robot can and can't do

A Kilobot is a small robot that can't see. It can:

- **measure the distance** to other robots within about 3 robot-widths (from infrared signal strength), and
- **talk** to them: broadcast a few numbers and hear theirs.

It can't see the shape, can't see the whole swarm, and has no GPS. The simulation follows the same limits: every decision uses only robots within `R_COMM`.

```python
G_DIST = 1.5        # robots this close count as neighbours for the gradient
R_COMM = 3.0        # how far a robot can talk to and measure other robots
```

All distances in this chapter are in **robot widths**, so 1.0 means two robots touching.

### 1.2 The four ingredients

| Ingredient | Question it answers | How |
| --- | --- | --- |
| **Seeds** | Where does the shape start? | 4 robots placed by hand, which know their positions |
| **Gradient** | How far am I from the seeds? | count the hops: one more than my closest neighbour's count |
| **Localization** | Where exactly am I? | trilateration from distances to robots that already know where they are |
| **Edge-following** | How do I get there? | travel clockwise around the outside of the group |

Then one rule decides where each robot stops. Together they build the shape **layer by layer outward from the seeds**.

### 1.3 Each robot is a state machine

Each robot is always in exactly one **state**, and simple rules move it from one state to the next:

```python
WAIT, MOVE_OUT, MOVE_IN, JOINED = 0, 1, 2, 3
```

| From | To | When |
| --- | --- | --- |
| Waiting | Moving outside | I have the highest gradient among the waiting robots near me, nobody near me is moving, and the gradient has had time to settle |
| Moving outside | Moving inside | I know my position, and it's inside the shape |
| Moving inside | Joined | my next step would leave the shape, **or** I'm touching a joined robot whose gradient is the same as mine or higher |

Joined robots never move again, and they become reference points that later robots use to find their own position. That's how the shape grows: every robot that joins helps the next one.

A **state machine** is a standard way to write behaviour like this: a small set of states, plus clear rules for switching between them. You'll see it again in later phases.

---

## Part 2: Setting up

### 2.1 Seeds and the starting cluster

```python
SEED_POS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [1.5, 0.866]])
SEED_GRADIENT = np.array([0, 1, 1, 1])
```

Four seeds sit where the shape begins. They know their positions (the shape's coordinates are defined relative to them) and start the gradient: the first seed has gradient 0, the other three have 1.

`initial_state` packs the movers into a hexagonal cluster to the left of the seeds, the way coins pack on a table, with neighbouring robots exactly 1 apart. 0.866 is the height of each row (√3 / 2) in that packing. Only the seeds start out localized; every other robot knows nothing about where it is.

### 2.2 The shape: a grid of yes/no cells

```python
RES = 0.25                          # each cell is a quarter of a robot wide
ORIGIN = np.array([-0.6, -0.6])     # bottom-left corner of the grid
GRID_SIZE = 200                     # 200 x 200 cells = 50 x 50 robot widths
AREA_PER_ROBOT = 1.2                # shape area to allow per robot (they don't pack perfectly)
```

The shape is stored as a 200 × 200 grid of `True`/`False` values, like a black-and-white image. Each robot carries this same grid: it's the "program" that says what to build. `make_shape` draws the star, wall or K into the grid and scales it so its area fits the number of robots. Robots don't pack perfectly when they stop, so it allows 1.2 square units per robot.

Checking whether a point is inside the shape means finding its cell:

```python
def inside_shape(p, mask):
    """Is point p inside the shape? Works inside jit and vmap."""
    idx = jnp.floor((p - ORIGIN) / RES).astype(jnp.int32)
    on_grid = jnp.all((idx >= 0) & (idx < GRID_SIZE))
    idx = jnp.clip(idx, 0, GRID_SIZE - 1)
    return on_grid & mask[idx[0], idx[1]]
```

Subtract the grid's corner, divide by the cell size, round down, and look the cell up. Points off the grid count as outside.

Note that a robot checks its **estimated** position against the shape, not its true position. It doesn't know its true position; it only knows what localization tells it.

---

## Part 3: The gradient

### 3.1 The idea

The gradient answers "how many robots away from the seeds am I?" Each robot sets its gradient to **one more than the smallest gradient among the robots touching it**. Robots touching a gradient-0 seed get 1, robots touching those get 2, and so on outward, like ripples in a pond.

No robot needs to know where the seeds are. The number spreads through the swarm one hop at a time, each robot only listening to its touching neighbours. This idea is called a **hop-count gradient**, and it's one of the oldest tools in swarm robotics.

### 3.2 In code

```python
    smallest = jnp.min(jnp.where(near, grad, MAX_GRADIENT))
    g = jnp.where(is_seed[i], grad[i], jnp.minimum(smallest + 1, MAX_GRADIENT))
```

`near` marks the robots within `G_DIST` (touching). `jnp.where(near, grad, MAX_GRADIENT)` keeps their gradients and replaces everyone else's with a huge number, so `jnp.min` finds the smallest gradient among touching neighbours only. Seeds keep their fixed values.

Two details matter:

- **It's recomputed from scratch every step.** When a robot leaves, its neighbours' gradients update on the next step. A robot doesn't remember its old value; it always listens to its neighbours now.
- **It spreads one hop per step.** A robot 18 hops from the seeds needs about 18 steps before its gradient is right. That's why robots wait `START_DELAY` (60) steps before anyone moves. (Exercise 3 shows what happens without the wait.)

### 3.3 Why the gradient matters

The gradient is used twice:

1. **Who leaves first:** the robots with the highest gradient, the far edge of the cluster, leave first. The cluster peels away from the far side, and nobody gets trapped.
2. **Where to stop:** a robot stops next to a joined robot with the same or higher gradient. That makes the shape fill in rings outward from the seeds (Part 6).

---

## Part 4: Localization by trilateration

### 4.1 The idea

A moving robot needs to know its position to check whether it's inside the shape. It knows its **distance** to nearby robots, and some of those robots (seeds and joined robots) know their **positions**. Is that enough?

- One distance: I'm somewhere on a circle around that robot.
- Two distances: I'm at one of the two points where two circles cross.
- **Three distances** (from robots not in a straight line): exactly one point fits all three.

Finding a position from distances to known points is called **trilateration**. GPS works the same way, using distances to satellites. That's why a robot needs at least 3 reference robots:

```python
    refs = in_comm & ~moving & localized
    can_localize = refs.sum() >= 3
```

References must be **stationary** (a moving robot's position estimate keeps changing) and **localized** (they must know where they are).

### 4.2 Refining a guess

Rather than solving the circle equations directly, the code starts from a guess and improves it, the same way the Kilobots did:

```python
def refine(p, refs, ref_est, dist):
    """Nudge guess p so its distance to each reference matches the measurement."""
    for _ in range(LOC_ITERS):
        vec = p - ref_est
        length = jnp.maximum(jnp.linalg.norm(vec, axis=1), 1e-6)
        # Where each reference "says" I should be: along the same line, at the measured distance.
        target = ref_est + vec / length[:, None] * dist[:, None]
        p = p + 0.5 * ((target - p) * refs[:, None]).sum(0) / jnp.maximum(refs.sum(), 1)
    return p
```

For each reference, draw a line from the reference through the current guess. The point on that line at the **measured** distance is where that reference "thinks" I am (`target`). Move the guess halfway toward the average of those targets, and repeat 8 times. If the guess is close, it settles on the true position after a few rounds.

### 4.3 Many starting guesses

A robot that has never been localized has no good first guess, and a bad one can settle on the wrong answer (for example, the mirror image across a line of references). So `localize` tries 9 starting guesses: the previous estimate, plus 8 points on a ring around the references. It refines each and keeps the one that fits the measurements best:

```python
def residual(p, refs, ref_est, dist):
    """How badly guess p disagrees with the measured distances (0 = perfect)."""
    err = jnp.linalg.norm(p - ref_est, axis=1) - dist
    return (refs * err ** 2).sum() / jnp.maximum(refs.sum(), 1)
```

The **residual** adds up how far off each measured distance is, squared so that positive and negative errors don't cancel. The best guess has the smallest residual. Running all 9 guesses at once is another use of `vmap`:

```python
    refined = jax.vmap(refine, in_axes=(0, None, None, None))(guesses, refs, ref_est, dist)
    errors = jax.vmap(residual, in_axes=(0, None, None, None))(refined, refs, ref_est, dist)
    return refined[jnp.argmin(errors)]
```

### 4.4 Errors build up

A joined robot's estimate becomes a reference for later robots, so any error gets passed on. In the real experiment, noisy distance sensing meant that errors built up and the far edges of shapes came out warped. In this simulation distances are exact, so errors stay small: about 0.05 robot widths on average. The small remaining error comes from the moving robot updating its estimate between fixes:

```python
    new_est = jnp.where(step_now, est_i + v, est_i)   # dead-reckoning guess until next fix
```

**Dead reckoning** means estimating where you are from how you've moved. The next trilateration fix corrects it.

---

## Part 5: Edge-following

### 5.1 The idea

Moving robots travel **clockwise around the outside of the whole group**: the cluster, the seeds and the growing shape. Picture walking around a building with your right hand on the wall. Even with no map, you'll eventually pass every part of the outside.

A robot does this by keeping a fixed distance from its **nearest stationary robot** and moving at right angles to it. As it passes one robot, a new one becomes the nearest, so it naturally follows the edge round corners.

### 5.2 In code

```python
    anchor_dist = jnp.where(in_comm & ~moving, dist, jnp.inf)
    j = jnp.argmin(anchor_dist)
    d_j = anchor_dist[j]
    has_anchor = jnp.isfinite(d_j)
    away = (p - pos[j]) / jnp.maximum(d_j, 1e-6)          # unit vector from anchor to me
    clockwise = jnp.array([away[1], -away[0]])            # 90 degrees to the right of "away"
    v = clockwise + K_RADIAL * (DESIRED - jnp.where(has_anchor, d_j, DESIRED)) * away
```

- `j` is the nearest stationary robot, the **anchor**. `jnp.inf` (infinity) hides moving robots from `argmin`.
- `away` is a unit vector from the anchor to me.
- Rotating a vector `(x, y)` by 90° clockwise gives `(y, −x)`. So `clockwise` is the direction to walk around the anchor with it on my right.
- The second term corrects my distance: if I'm closer than `DESIRED` (1.0), it pushes me away; if I'm further, it pulls me in. `K_RADIAL` sets how hard.

Then a push away from any robot closer than `TOO_CLOSE` stops overlaps, and the velocity is scaled to exactly `SPEED`.

The real Kilobots couldn't move sideways like this; they could only drive forward and turn. They followed the edge by turning toward the group when too far and away when too close. The simulated version reaches the same result more directly.

### 5.3 Taking turns

Two rules stop movers from crashing into each other:

```python
    quiet = ~jnp.any(in_comm & moving)
```

A waiting robot only starts if no robot within `R_COMM` is moving, which spaces out departures.

```python
    earlier = moving & ((start_time < start_time[i]) | ((start_time == start_time[i]) & (ids < i)))
    blocked = jnp.any(others & earlier & (dist < YIELD_DIST))
```

A mover pauses while a robot that started **earlier** is within `YIELD_DIST`, like cars queueing. Because "earlier" is a strict order (ties broken by id), the first robot in any queue never waits, so the queue can never jam completely.

---

## Part 6: Joining the shape

```python
    entered = (state[i] == MOVE_OUT) & loc_i & inside_shape(est_i, mask)
    new_state = jnp.where(entered, MOVE_IN, state[i])

    leaving = ~inside_shape(est_i + v, mask)
    stopped_nearby = near & ((state == JOINED) | is_seed)
    caught_up = jnp.any(stopped_nearby & (grad >= g))
    stop = (new_state == MOVE_IN) & (leaving | caught_up)
```

Once inside the shape, a robot keeps edge-following until one of two things happens:

1. **`leaving`:** its next step would take it outside the shape. It stops at the edge, so the shape's outline gets filled.
2. **`caught_up`:** it's touching a joined robot whose gradient is the same as its own or higher. That means it has reached the current outer layer of the growing shape, so it stops there and becomes part of it.

Rule 2 is what makes the shape fill solidly, ring by ring, instead of only around its outline. Exercise 2 shows what happens without it.

---

## Part 7: Making the simulation fast enough

### 7.1 Two parts per robot

Each robot's program is split into two functions:

- **`decide_one` (every robot):** update my gradient, and decide whether to start moving. This is cheap.
- **`move_one` (moving robots only):** localize, edge-follow, and decide whether to stop. Trilateration with 9 guesses makes this expensive.

On real robots it doesn't matter: every robot has its own chip and runs everything. In the simulation one CPU does all the work, and at most 14 robots move at the same time. So `step` only runs `move_one` for the movers:

```python
    slots = jnp.argsort(~moving)[:MAX_MOVERS]
    real = moving[slots]
```

`~moving` is `False` for movers and `True` for everyone else, and `argsort` puts `False` before `True`, so the movers' indices come first. Taking the first `MAX_MOVERS` (24) gives a fixed-size list of slots, which JAX needs, with `real` marking which slots hold actual movers. After `move_one` runs on the slots, `.at[slots].set(...)` writes the results back. Slots that aren't real movers write back their old values unchanged.

This made the simulation about 3× faster without changing the result at all. The checks confirm that no more than 24 robots ever move at once.

### 7.2 Saving frames, not every step

Assembly takes about 6,500 steps. Saving every step would make huge arrays, so `simulate` uses two nested `scan`s: an inner one that runs 20 steps, and an outer one that saves one frame after each block of 20. You get 350 frames instead of 7,000.

---

## Part 8: What the checks prove

| Check | What a failure would mean |
| --- | --- |
| Gradient equals hop count from the seeds | the gradient rule is wrong; compared against breadth-first search, a standard way to count hops in a network |
| Every mover joins the shape | robots got stuck circling or waiting |
| The shape is at least 85% covered | the shape filled with gaps |
| No two robots overlap | collision avoidance failed |
| Joined robots know where they are | localization drifted |
| Joined robots are inside the shape | robots stopped in the wrong places |
| Joined robots never move again | the state machine let a joined robot move |
| Movers never exceed `MAX_MOVERS` | some movers were skipped by the speed-up |
| A robot outside `R_COMM` doesn't affect a mover | a decision secretly uses non-local information |
| A robot within `R_COMM` does affect a mover | the robot isn't really sensing its neighbours |
| Star and K assemble | the algorithm only works for one shape |

### What a real bug looked like

The first version left 3 robots waiting forever, right next to the seeds. Their gradient tied with an already-joined robot, and they lost the tie-break, so they never had "the highest gradient around" and never started. The fix: compare gradients only against other **waiting** robots, because joined robots aren't competing to leave. Debugging usually goes like this: notice something odd, look at exactly which robots are affected and what they can see, and find the rule that doesn't match the intent.

---

## How this differs from the real experiment

Honest differences, worth knowing before you talk to anyone about this:

| Real Kilobots (2014) | This simulation |
| --- | --- |
| 1,024 robots, about 12 hours | 150 robots, about 6,500 steps |
| Noisy distance sensing from infrared signal strength | exact distances |
| Drive forward and turn; slow and imprecise | move directly in any direction |
| Robots run at their own pace, not in lockstep | every robot updates once per step, at the same time |
| Robots sometimes got stuck or collided, needing human help | none of that, apart from the bugs fixed above |

The algorithm (gradient, trilateration, edge-following, and the joining rule) is the same idea. The simulation is cleaner than reality, which is one reason the real paper's result is so impressive.

---

## Try this

Each was tested; predict first, then run `python phase2/animate.py` or `python phase2/checks.py`. Undo each change afterwards.

1. Run each shape: `--shape wall`, `--shape star`, `--shape k`. Which one fills least completely, and where are its gaps?
2. In `move_one`, change `stop = (new_state == MOVE_IN) & (leaving | caught_up)` to `stop = (new_state == MOVE_IN) & leaving`. Predict what the shape looks like, then watch.
3. Set `START_DELAY = 0`. Does the shape still assemble? Run `checks.py`: which check fails, and why might robots that start before the gradient settles cause it?
4. Set `YIELD_DIST = 0.0` (no queueing). Does anything go wrong? If not, which other rule is already keeping movers apart?
5. In `move_one`, change `refs.sum() >= 3` to `refs.sum() >= 2`. What happens to the robots that localize from only two references?
6. Add your own shape: write a function like `_wall` (for example, a ring or a plus sign), add it to `SHAPES`, and run it.
7. Harder: make distance sensing noisy, like the real Kilobots. In `move_one`, add a small random error to `dist` before it's passed to `localize`. You'll need a random key. How does the localization error check change?

---

## Check your understanding

<details>
<summary>1. What is the gradient, and how does each robot compute it?</summary>

Its hop count from the seeds. Each robot sets its gradient to one more than the smallest gradient among the robots touching it (within `G_DIST`). Seeds keep fixed values (0 and 1). The values spread outward one hop per step.
</details>

<details>
<summary>2. Why do robots wait <code>START_DELAY</code> steps before anyone moves?</summary>

The gradient spreads one hop per step, so far-away robots don't have correct gradients at first. Robots use the gradient to decide who leaves first, so moving before it settles means the wrong robots set off.
</details>

<details>
<summary>3. Why does localization need at least 3 reference robots?</summary>

One distance puts you on a circle; two distances leave two possible points where the circles cross; three distances, from references not in a straight line, leave only one point that fits.
</details>

<details>
<summary>4. Why must references be stationary and localized?</summary>

They're used as known positions. A moving robot's position keeps changing, and an unlocalized robot doesn't know its position at all, so neither can serve as a fixed point to measure from.
</details>

<details>
<summary>5. Why does <code>localize</code> try 9 starting guesses?</summary>

Refinement from a bad starting guess can settle on a wrong answer, such as a mirror image. Trying several starting points and keeping the one with the smallest residual (the best fit to the measured distances) makes it robust for robots localizing for the first time.
</details>

<details>
<summary>6. How does edge-following work with no map?</summary>

The robot keeps a fixed distance from its nearest stationary robot and moves at 90° to it (clockwise). As it passes one robot, the next becomes the nearest, so it traces the outside of the group, like keeping your right hand on a wall.
</details>

<details>
<summary>7. What are the two stopping rules, and what does each one do for the shape?</summary>

Stop if the next step would leave the shape: this fills the outline. Stop when touching a joined robot with the same or higher gradient: this makes robots stop at the current outer layer, so the shape fills solidly, ring by ring, instead of only around its edge.
</details>

<details>
<summary>8. Why is the robot's program split into <code>decide_one</code> and <code>move_one</code>?</summary>

To make the simulation faster. Every robot needs the cheap part (gradient and the start decision), but only moving robots need the expensive part (trilateration and motion). Running `move_one` only for at most 24 movers made the simulation about 3× faster with identical results. Real robots wouldn't need this, since each has its own chip.
</details>

<details>
<summary>9. Why can't the movers' queue jam completely?</summary>

A mover only waits for movers that started earlier, and "earlier" is a strict order (ties broken by id). The first robot in any queue has nobody earlier to wait for, so it always keeps moving.
</details>

<details>
<summary>10. Name two ways this simulation is easier than the real Kilobot experiment.</summary>

Any two of: exact distance measurements instead of noisy infrared; direct movement in any direction instead of drive-and-turn; all robots updating in lockstep; 150 robots instead of 1,024; no hardware failures.
</details>

---

## Glossary

| Term | Meaning |
| --- | --- |
| Self-assembly | Parts arranging themselves into a structure with no outside builder |
| Seed | A robot placed by hand that knows its position and starts the gradient |
| Hop-count gradient | Each robot's distance from the seeds, counted in robot-to-robot hops |
| Trilateration | Finding a position from distances to known points |
| Residual | How badly a guess disagrees with the measurements; smaller is better |
| Dead reckoning | Estimating position from your own movement since the last fix |
| Edge-following | Travelling around the outside of a group by keeping a fixed distance from the nearest robot |
| State machine | Behaviour defined by a few states and rules for switching between them |
| Anchor | The nearest stationary robot a mover is circling |
| Breadth-first search | Exploring a network outward one hop at a time; used to check the gradient |
