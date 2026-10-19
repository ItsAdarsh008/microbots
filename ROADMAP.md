# Microbot Swarm Demo — Roadmap

Oct 14, 2026 · @Adarsh

This is the plan for a self-healing 3D microbot swarm built in JAX: every bot runs the same small program, with no central brain. Phase 0 is in progress. The code lives at [github.com/ItsAdarsh008/microbots](https://github.com/ItsAdarsh008/microbots).

How to use this doc: read the Overview and timeline first, then work from the current phase. Tick checklist items as they're done, and leave a comment on any line you have a question about.

## Overview

The goal is a self-healing, environment-aware 3D microbot swarm in about 12 weeks at roughly 8–10 hours a week, about 85–110 hours total. Every bot acts on its own local view, and a Jev decision layer turns commands into shapes.

The demo you ship:

- Thousands of bots in 3D, each sensing only neighbours within a small radius. No bot knows the global layout. You learn the basics in 2D (Phases 0–2), and everything from Phase 3 onward is 3D.
- On command, the swarm builds a wall, bridge, ring or letter. Given terrain it has never seen (a gap, obstacles, anchor points), it grows a new structure that fits, not a stored template.
- Click to delete a chunk of bots, and the rest regrow the shape.
- Typed or spoken commands go to Jev, which picks the target shape and only acts above a confidence threshold. This stands in for the movie's headband.

The architecture has two layers:

- **Local layer (every tick, on your laptop):** JAX runs the same small program on every bot in parallel. It starts as hand-written rules, then becomes the Kilobot algorithm, then a learned neural cellular automaton.
- **Decision layer (occasionally, in the cloud):** Jev answers typed questions such as which shape a command means, or whether a cluster should hold, reform or merge. Each answer comes with a calibrated confidence.

The estimates assume a busy first-year schedule. If you have more time in a week, the phases compress; they don't need to run in strict order once phase 1 works.

## Timeline at a glance

| Phase | Hours | Weeks |
| --- | --- | --- |
| 0 · Setup and JAX basics | 6–8 h | 1 |
| 1 · Local-rule swarm | 8–10 h | 1–2 |
| 2 · Kilobot shapes | 12–15 h | 2–4 |
| 3 · 3D self-healing NCA | 18–24 h | 4–7 |
| 3.5 · Environment-aware NCA (3D) | 15–22 h | 7–9 |
| 4 · Jev decision layer | 5–7 h | 9–10 |
| 5 · Polish and visuals | 8–10 h | 10–11 |
| 6 · Package and post | 6–8 h | 11–12 |

Milestones: 1,000-bot swarm (end of week 2), first shape GIF (week 4), 3D self-heal (week 7), new terrain (week 9), ship (week 12). Phases 0–2 are 2D; Phase 3 onward is 3D.

Live version with the timeline chart: https://claude.ai/code/artifact/83d6998b-55b6-41dc-b691-dde7a5b44f1b

Start reading for the next phase while you finish the current build; that overlap is what keeps the total near 12 weeks.

## Phase-by-phase plan

Each phase ends with something you can show, so you have a working demo early and keep improving it. Each phase also closes with a checklist that combines the build, posting and outreach steps, so you can tick off everything before moving on.

Each phase also has a **Post** line (what to share publicly) and a **Reach out** line (who to contact once that phase gives you something to show). Each step up in outreach waits until you have a clip that proves the previous phase.

Outreach rules for every DM or email:

- Lead with the clip or GIF, not your background.
- Say in one line what you built and what you borrowed from their work.
- Ask one specific question they can answer in two minutes. Don't ask "can we chat?"
- Follow up once after 7–10 days, then let it go.
- Log every contact (name, date, what you sent, reply) so later phases can follow up with new results.

### Phase 0 — Setup and foundations · 6–8 h · week 1

**Learn:** NumPy array thinking (vectorizing instead of looping), JAX basics (`jit`, `vmap`, pure functions, `jax.random` keys), and plotting with Matplotlib animation or Pygame.

**Build:** 100 dots doing a random walk inside a box, animated, run with `vmap` over every dot.

**Done when:** you can explain why JAX needs explicit random keys and pure functions.

**Post:** a kickoff post ("Building Big Hero 6-style microbots in JAX, follow along") with the random-walk clip. Saying it publicly commits you to finishing.

**Reach out:** no DMs yet. Follow 10–15 people working on swarm robotics, neural cellular automata and JAX, and leave thoughtful comments on their posts so your name is familiar later.

#### Phase 0 walkthrough: build it yourself

You'll build the random walk twice: once in NumPy to get the idea, then in JAX to learn the tools the rest of the project depends on. Each step ends with a check so you know it works before moving on. This section explains the ideas and the functions to look up, but leaves the code to you. It assumes only basic Python (variables, functions and loops), and every new idea is explained the first time it appears.

**Step 1 — Setup (about 30 min)**

1. Install the Python Install Manager for Windows from python.org/downloads or the Microsoft Store (not the source tarball, which has to be compiled). Then run `py install 3.13` in a terminal.
2. Make a project folder, open PowerShell in it (in Git Bash, activate with `source .venv/Scripts/activate` instead), and create a virtual environment: `py -V:3.13 -m venv .venv`, then activate it with `.venv\Scripts\activate`. Your prompt should now start with `(.venv)`. If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
3. Install the libraries: `pip install jax numpy matplotlib pillow`.
4. Run `git init`, add a `.gitignore` containing `.venv/`, and push to a public GitHub repo now. Your commit history becomes part of the proof of work.

**Project repo:** the code lives at [github.com/ItsAdarsh008/microbots](https://github.com/ItsAdarsh008/microbots), which already ignores `.venv/`. Collaborators should clone it with `git clone https://github.com/ItsAdarsh008/microbots` and work in that folder instead of running `git init`.

**Check:** `python -c "import jax; print(jax.devices())"` prints a CPU device.

**What each Step 1 command means**

- `py install 3.13`: tells the Python Install Manager to download and install Python 3.13.
- `py -V:3.13 -m venv .venv`: `py` is the launcher, `-V:3.13` picks Python 3.13, `-m venv` runs Python's built-in virtual-environment tool, and `.venv` is the folder it creates. That folder holds a private copy of Python just for this project.
- `.venv\Scripts\activate`: switches this terminal to the project's private Python. While `(.venv)` shows at the start of the prompt, `python` and `pip` mean the project's copies. You need to run it again in every new terminal.
- `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`: Windows blocks scripts by default, including `activate`. This allows scripts made on your own computer to run, for your user account only. It's only needed once, and only if activate gives an error.
- `pip install jax numpy matplotlib pillow`: `pip` is Python's package installer. It downloads JAX (the simulation engine), NumPy (arrays), Matplotlib (plotting and animation) and Pillow (used to save GIFs) into `.venv`.
- `git init`: turns the folder into a Git repository so every change can be saved as a commit.
- `.gitignore` containing `.venv/`: tells Git to skip the `.venv` folder. It's large, specific to your computer and can always be rebuilt, so it doesn't belong on GitHub.
- `python -c "import jax; print(jax.devices())"`: `-c` runs the quoted text as a tiny Python program without making a file. It loads JAX and prints the hardware JAX can use. Seeing a CPU device listed means JAX installed correctly; an error means something in the earlier steps went wrong.

**Step 2 — NumPy version (45–60 min)**

The ideas:

**1. The swarm is stored as one table of numbers, called an array.**

Each bot has a position: an x value (left–right) and a y value (up–down). Instead of making a separate variable for each bot, you store every bot's position in one table with a row per bot and two columns, x and y. NumPy calls this table an array. Its *shape* tells you its size: `(100, 2)` means 100 rows (bots) and 2 columns (x and y).

```python
import numpy as np
pos = np.random.uniform(0, 10, size=(100, 2))  # 100 bots at random spots in a 10 × 10 box
print(pos.shape)  # (100, 2)
print(pos[0])     # the first bot's x and y
```

**2. Change every bot at once instead of looping over them.**

Looping means handling one bot, then the next, then the next. With an array, you write one rule and it applies to every row at once, like writing "column A = column A + 1" in a spreadsheet instead of editing each cell by hand.

```python
# Loop: one bot at a time (slow, avoid this)
for i in range(100):
    pos[i] = pos[i] + 1

# Array: every bot at once (fast, do this)
pos = pos + 1
```

Both give the same result. The array version runs 50–100 times faster in Python because NumPy hands the whole table to fast, pre-built code. JAX's speed features only work with this style, so whenever you're about to write "for each bot…", stop and write it as one rule for the whole table.

**3. A random walk: every step, each bot moves a small random amount.**

`np.random.normal(0, 0.1, size=(100, 2))` makes a table the same shape as `pos`, filled with random numbers. They come from a *normal distribution*: most are close to 0, a few are bigger, and they're equally likely to be positive or negative. The 0.1 controls how big the moves are. Adding this table to `pos` nudges every bot in a random direction.

**4. Walls keep the bots inside the box.**

After a move, some bots end up past an edge of the box, which runs from 0 to L (10 in the example above). You have two options:

- **Clamp:** `np.clip(pos, 0, L)` pushes any bot outside back onto the edge. It's simple, but bots end up stuck along the walls.
- **Reflect:** bounce the bot back by the amount it overshot, so a bot at x = −0.3 goes to x = 0.3, and one at x = L + 0.3 goes to L − 0.3. This looks more natural.

For reflecting, use `np.where(condition, value_if_true, value_if_false)`. It checks the condition for every number in the table and picks one of the two values for each, so it works like an if statement for a whole array. For example, `pos = np.where(pos < 0, -pos, pos)` handles the left and bottom walls.

**5. Velocity makes the motion smooth (optional, but it looks much better).**

Pure random nudges look like jitter, because each step ignores the one before. Give each bot a velocity as well: another `(100, 2)` table saying how far it moves per step. Each step, set velocity = 0.9 × velocity + a small random nudge, then position = position + velocity. The 0.9 means each bot mostly keeps going the way it was going, so it drifts in smooth curves. When a bot bounces off a wall, flip the sign of that part of its velocity too, so it actually heads away from the wall.

**6. Frames: saving each step so you can animate it.**

An animation is a series of pictures, called frames. After every step, add a copy of `pos` to a Python list. After 300 steps, `np.stack(frames)` glues them into one array of shape `(300, 100, 2)`: frame number, then bot, then x and y. For example, `traj[50, 3]` is bot 3's position at step 50.

What to write: a function `step(pos, vel)` that returns the new position and velocity, and a loop that runs it 300 times and stores every frame. Stack the frames with `np.stack`.

**Check:** the stacked result has shape `(300, 100, 2)`, and every value is between 0 and L.

**Step 3 — Port to JAX (2–3 h, the real learning)**

JAX looks like NumPy but follows stricter rules. Those rules are what make it fast and parallel, and you'll rely on them in every later phase.

**1. `jax.numpy` is a near copy of NumPy.**

Import it with `import jax.numpy as jnp`, then change `np.` to `jnp.` in your Step 2 code: `np.where` becomes `jnp.where`, and so on. Most of your code carries over unchanged.

**2. JAX arrays can't be edited in place.**

In NumPy you can change one cell directly, for example `pos[0, 0] = 5`. JAX doesn't allow changing an existing array; you always make a new one instead. Usually this is no extra work, because `pos = pos + vel` already creates a new array. To change one cell, write `pos = pos.at[0, 0].set(5)`, which returns a new array with that cell changed. JAX works this way so it can safely reorder and parallelize your calculations.

**3. Random numbers need a "key".**

Computers can't make truly random numbers. They run a formula, starting from a number called a seed, that produces numbers which look random. NumPy hides this and keeps track of the seed for you. JAX makes you hold it yourself, as a *key*:

```python
import jax
key = jax.random.PRNGKey(0)          # 0 is the seed
key, subkey = jax.random.split(key)  # make a fresh key to use now
noise = jax.random.normal(subkey, shape=(100, 2))
```

The same key always gives the same numbers. So each step, split off a new subkey and use that. The rule to remember: never use the same key twice.

**4. Functions must be "pure".**

A pure function works like a maths function: the same inputs always give the same output, and it does nothing else. It doesn't change variables outside itself, and it doesn't use hidden randomness. That's why the key is passed into your step function as an argument instead of being created inside it. JAX needs this guarantee to compile and parallelize your code safely.

**5. `jax.jit` compiles a function for speed.**

Normally Python runs your code one line at a time, which is slow. `fast_step = jax.jit(step)` makes JAX read the whole function once, turn it into optimized machine code, and reuse that every time you call it. The first call is slow because that's when the compiling happens; later calls are much faster. One catch: while compiling, JAX only knows the shapes of your arrays, not the actual numbers, so a Python `if pos[0, 0] < 0:` inside the function fails. Use `jnp.where` instead.

**6. `jax.vmap`: write it for one bot, run it for every bot.**

Sometimes it's easier to think about a single bot: "here's my position, my velocity and my random key; what's my new position?" Write a function `step_one` that handles one bot, where its position has shape `(2,)`. Then `step_all = jax.vmap(step_one)` creates a version that takes the whole `(100, 2)` table and runs `step_one` on every row at the same time. Each bot needs its own key, so make 100 with `keys = jax.random.split(key, 100)`. This is the core idea of the whole project: every bot runs the same small program on its own. Today plain array maths would work too, but once bots make decisions based on their neighbours, writing the logic for one bot is much easier.

**7. Stretch: `jax.lax.scan` replaces the Python loop.**

Your Step 2 loop runs the step function 300 times in Python. `jax.lax.scan` does the repeating inside compiled code, which is faster. You give it a function that takes the current state (called the *carry*: positions, velocities and key) and returns the new state plus anything to save, such as this frame's positions. It returns the final state and all the saved frames stacked together. Skip this if it's confusing; a Python loop is fine for Phase 0.

**Checks:**

- The output shape matches your NumPy version.
- Running twice with the same seed gives identical results, and a different seed gives different ones.
- With N = 10,000, time the jitted step against the plain one. Call it once first to compile, and add `.block_until_ready()` so the timer waits for JAX to actually finish (JAX hands control back before the work is done).

**Step 4 — Animate and record (about 45 min)**

- **Simulate first, draw after.** Run all 300 steps and save the frames, then animate them. Matplotlib can't read JAX arrays directly, so convert them first with `traj = np.asarray(traj)`.
- **How the animation works.** Matplotlib draws one picture, called a figure. `FuncAnimation` calls a function you write, `update(frame)`, once per frame. Your function moves the dots to that frame's positions, and Matplotlib turns the pictures into a video. The pieces you'll need:
    - `dots = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=4)` draws frame 0. `traj[0, :, 0]` means "frame 0, every bot, column 0 (x)"; `s=4` sets the dot size.
    - Inside `update(frame)`, `dots.set_offsets(traj[frame])` moves every dot to that frame's positions.
    - `anim = FuncAnimation(fig, update, frames=300, interval=33)` runs it, showing each frame for 33 milliseconds, which is about 30 frames per second.
- **For a clean look:** `plt.style.use("dark_background")`, hide the axes with `ax.axis("off")`, use small dots, set the limits to the box with `ax.set_xlim(0, L)` and `ax.set_ylim(0, L)`, and keep it square with `ax.set_aspect("equal")`.
- **Saving:** `anim.save("random_walk.gif", writer="pillow", fps=30)` writes a GIF. An MP4 looks better on X but needs ffmpeg, a free video tool you install separately. You can also screen-record the live window with OBS.
- **For the kickoff post,** aim for 8–12 seconds and try N = 1,000 to show scale.

**Check:** the clip plays smoothly and no dot ever leaves the box.

**Common bugs**

- Every bot moves identically: one key was used for all of them instead of a split per bot.
- Same "random" numbers every step: the key isn't being split between steps.
- `ConcretizationTypeError`: a Python `if` on an array inside a jitted function. Switch to `jnp.where`.
- Randomness frozen after the first call: `np.random` was used inside a jitted function.
- Confusing errors in general: print `.shape` of everything. Most bugs are shape bugs.

**Done when you can answer these in your own words** (each makes a good "today I learned" post):

- Why does JAX make you pass random keys instead of using a global seed?
- What happens on the first call to a jitted function, compared with later calls?
- What does vmap do to the input shapes of `step_one`?
- Why can't you use a Python `if` on an array inside a jitted function?

**Docs to keep open:** [Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), [Pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html), [Automatic vectorization](https://docs.jax.dev/en/latest/automatic_vectorization.html) and [Matplotlib FuncAnimation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html).

**Phase 0 checklist**

- [ ] Python, virtual environment and libraries installed; `jax.devices()` prints a CPU
- [ ] Public GitHub repo created and first commit pushed
- [ ] NumPy random walk working (shape and bounds checks pass)
- [ ] JAX version using keys, `jit` and `vmap` (all three checks pass)
- [ ] Random-walk clip recorded (8–12 s)
- [ ] Can answer the four done-when questions in your own words
- [ ] X profile set up: bio written, 100–150 accounts followed, notifications on for 10
- [ ] Kickoff post on X with the clip, pinned (weekday, 8–10 AM Eastern)
- [ ] LinkedIn kickoff post and Instagram story
- [ ] Following 10–15 people in swarm robotics, NCA and JAX, with a few thoughtful replies left
- [ ] At least 3 "today I learned" posts

### Phase 1 — A swarm with local rules · 8–10 h · weeks 1–2

**Learn:** neighbour search (a uniform grid or cell list, not all-pairs), the Boids rules (separation, alignment, cohesion), and simple collision handling.

**Build:** 1,000 bots that flock, avoid each other and move toward a target region, each seeing only neighbours within radius r.

**Done when:** the swarm runs at 30+ FPS at 1,000 bots on your laptop, and the rules contain no global information.

**Post:** the 1,000-bot flocking clip, plus a short thread on what you learned about neighbour search and why it took FPS from X to Y.

**Reach out:** join communities rather than messaging individuals. Good places are the JAX GitHub Discussions, r/robotics, and a UWaterloo robotics or ML club. Ask one technical question in public.

**Phase 1 checklist**

- [ ] Neighbour search using a grid or cell list
- [ ] Separation, alignment and cohesion working
- [ ] 1,000 bots at 30+ FPS, with no global information in the rules
- [ ] Flocking clip posted on X with a "no leader" hook
- [ ] Mini-thread on neighbour search with real FPS numbers
- [ ] Quote-posted at least one relevant demo or paper with your own angle
- [ ] Joined JAX GitHub Discussions, r/robotics and a UWaterloo club, and asked one technical question in public

### Phase 2 — Kilobot-style shape formation · 12–15 h · weeks 2–4

**Learn:** the three behaviours from Rubenstein et al. (2014): a hop-count gradient from seed bots, local position estimation by trilateration, and edge-following until a bot is inside the shape. Also finite state machines per bot.

**Build:** the swarm self-assembles into a star, a wall and a letter from a bitmap. Add a shape switcher on keys 1–4.

**Done when:** you get a clean GIF of the swarm forming a shape. This is your first postable result.

**Post:** the first shape-formation GIF next to the matching figure from the Kilobot paper, crediting the authors.

**Reach out (credibility checkpoint 1):** you now have a faithful reimplementation of a well-known paper.

- Email or DM the Kilobot authors (Michael Rubenstein and Radhika Nagpal) and one or two programmable-matter researchers (MIT's M-Blocks group, CMU's Claytronics). Send the GIF and one question about where your version breaks.
- Email 2–3 UWaterloo professors working on robotics or multi-agent systems. Ask whether they take first-year undergraduate research assistants, and attach the GIF.

**Phase 2 checklist**

- [ ] Hop-count gradient from seed bots
- [ ] Position estimates by trilateration
- [ ] Edge-following state machine
- [ ] Swarm forms a star, a wall and a letter; keys 1–4 switch shapes
- [ ] Clean shape-formation GIF
- [ ] X thread: the paper's figure next to your version, authors credited
- [ ] LinkedIn milestone post and first Instagram reel
- [ ] Pinned post updated to the best clip so far
- [ ] Contact log started (name, date, what you sent, reply)
- [ ] Checkpoint 1: messaged the Kilobot authors and 1–2 programmable-matter researchers
- [ ] Checkpoint 1: emailed 2–3 UWaterloo professors about undergraduate research roles

### Phase 3 — 3D neural cellular automaton, self-healing · 18–24 h · weeks 4–7

**Learn:** convolutions as "look at my neighbours", training loops with Flax and Optax, the Growing NCA tricks (sample pool, damage during training, alive masking), the jump from a grid NCA to mapping cells onto bots, and then the move to 3D: a voxel grid (a 3D grid of cubes, like pixels with depth) and 3D convolutions, where each cell looks at the 26 neighbours in the 3×3×3 cube around it.

**Warm-up (adds 4–6 h, worth it):** build a small CNN from scratch in pure JAX, for example an MNIST classifier with no Flax. An NCA is a tiny CNN, so this makes the rest of the phase much easier. Reference: [CS231n notes on CNNs](https://cs231n.github.io/convolutional-networks/). Post the result as "CNN from scratch in 100 lines of JAX".

**Build:** first, a 2D NCA in JAX that grows and regenerates a target shape (6–8 h, following the Distill article). Then extend it to a 3D voxel NCA on a 32×32×32 grid that grows a 3D target such as a tower, arch or bridge (8–12 h). Train the 3D version on a free Google Colab GPU, because it's too slow on a laptop CPU. Finally, render each live cell as a microbot in a 3D viewer that steers toward its cell, and add click-to-damage (3–4 h).

**Done when:** you carve 30% out of the 3D shape and it regrows in a few seconds.

**Why 3D starts here:** the 2D NCA teaches you the method on a small, fast problem, and the 3D version is the same idea with one more dimension. Getting 2D working first means that when 3D breaks, you know the bug is in the 3D part.

**Reference for 3D:** [Sudhakaran et al., Growing 3D Artefacts and Functional Machines with Neural Cellular Automata (2021)](https://arxiv.org/abs/2103.08737) grows 3D structures with NCAs and shows them regenerating after damage. Read it after the Distill article.

**Post:** the 3D self-healing clip, with the camera slowly orbiting. This is your strongest single post, so make it the hook: delete a chunk and watch it regrow. Follow it with a short write-up of what made training work.

**Reach out (credibility checkpoint 2):** you now have an original result, not just a reproduction.

- Share it with the Growing NCA authors (Alexander Mordvintsev's group at Google) and researchers working on collective or emergent intelligence, for example at Sakana AI.
- Follow up with everyone from checkpoint 1 who replied, using the new clip.
- Apply: ML or research co-op postings for your first work term, and undergraduate research positions at UWaterloo. Link the clip in the first line of every application.

**Phase 3 checklist**

- [ ] Warm-up: CNN from scratch in pure JAX
- [ ] "CNN from scratch" mini-thread posted
- [ ] 2D NCA grows the target shape (Distill replication)
- [ ] 3D voxel NCA, trained on Colab, grows a 3D target and regrows within a few seconds after 30% damage
- [ ] 3D cells rendered as bots in a 3D viewer, with click-to-damage
- [ ] Self-healing clip posted on X at a time you can stay online for 2 hours, then pinned
- [ ] Follow-up thread on what made training work, with the repo linked in the replies
- [ ] LinkedIn milestone post and Instagram reel
- [ ] Checkpoint 2: shared with the Growing NCA authors and emergent-intelligence researchers
- [ ] Checkpoint 2: followed up with everyone who replied at checkpoint 1
- [ ] Checkpoint 2: applied to ML or research co-op postings and UWaterloo research positions

### Phase 3.5 — Environment-aware NCA · 15–22 h · weeks 7–9

The swarm stops copying templates and grows structures that fit terrain it has never seen. This turns the demo from "switches between shapes" into "adapts to its environment".

**Learn:** extra input channels the network reads but can't overwrite (terrain, obstacles, anchor points), procedural environment generation, losses that score what a structure does rather than what it looks like, and testing on environments held out from training.

**Build, in two steps:**

1. **Procedural training (8–10 h):** generate thousands of random 3D environments with gaps of different widths, pillars, walls and pairs of anchor points. Compute a target for each with a simple rule, such as the shortest thick path between the anchors that avoids obstacles. Train the NCA with the terrain as fixed input channels, then test it on layouts it never saw.
2. **Functional loss (4–8 h):** replace the target picture with a score: anchors connected, obstacles avoided, as few bots used as possible. With no picture to copy, the NCA works out its own structure for each environment. This is the step that produces shapes outside any template.

**Done when:** you draw a new obstacle course or widen the gap live, and the swarm grows a different structure that still connects the anchors, including after you damage it.

**The limit to state honestly:** it invents structures only for the kinds of goals it was trained on, such as connect, block or enclose. A brand-new goal needs new training; open-ended goals are the multi-agent RL stretch goal.

**Reference:** [Sudhakaran et al., Goal-Guided Neural Cellular Automata (2022)](https://arxiv.org/abs/2205.06806) shows NCAs steered by an external goal signal, which is close to what this phase adds.

**Post:** a clip where you widen the gap and the bridge re-grows to fit, then add an obstacle and it routes around. Caption: "No templates. It has never seen this layout."

**Reach out:** this extends the Growing NCA work, so follow up with any NCA researchers who replied at checkpoint 2, plus the Goal-Guided NCA authors. Ask whether the functional-loss result is worth a short workshop paper; their answer tells you how original it is.

**Phase 3.5 checklist**

- [ ] Procedural 3D environment generator (gaps, pillars, walls, anchor points)
- [ ] NCA trained with terrain input channels and working on held-out layouts
- [ ] Functional-loss version connects anchors with no target picture
- [ ] Live demo: widening the gap or adding an obstacle makes it adapt
- [ ] "No templates" clip posted on X
- [ ] Side-by-side post: template NCA against environment-aware NCA
- [ ] Ran at least 3 obstacle courses suggested by followers
- [ ] Quote-posted the Goal-Guided NCA paper with your result
- [ ] Followed up with NCA researchers and the Goal-Guided NCA authors about a workshop paper

### Phase 4 — Jev decision layer · 5–7 h · weeks 9–10

**Learn:** calling a hosted model API, Jev's typed questions (Choice and Score) and confidence thresholds, and running API calls asynchronously so the simulation never freezes.

**Build:** a text box (and optionally speech-to-text) where you type "make a bridge". Jev returns a Choice from your shape list plus a confidence, and the swarm only switches above a threshold such as 0.8. Optional: squad leaders summarize their cluster in text and Jev picks `hold`, `reform` or `merge`.

**Done when:** a vague command such as "get us across the gap" maps to the bridge shape, and nonsense gets rejected.

**Post:** "A swarm you talk to: a typed decision model picks the formation, and only acts above 0.8 confidence." Include the accuracy numbers and tag TypeSafe.

**Reach out (credibility checkpoint 3):** you are now one of the few people with a public, non-text demo built on Jev.

- Send the TypeSafe team the clip and your command-accuracy numbers. Offer a short case study or a bug report from the failure cases, and ask whether they take part-time or intern help.
- Apply to agent and decision-model startups on your part-time list, leading with this demo as proof you can ship a hybrid ML system.

**Phase 4 checklist**

- [ ] Jev access set up through OpenRouter or Vercel AI Gateway, with the key in `.env`
- [ ] API calls run asynchronously and the sim never freezes
- [ ] Commands map to shapes, gated at 0.8 confidence, and nonsense is rejected
- [ ] Accuracy measured on 20 test commands, Jev against keyword matching
- [ ] Clip posted with the confidence score on screen, TypeSafe tagged
- [ ] Post on decision models versus LLMs for control
- [ ] LinkedIn milestone post
- [ ] Checkpoint 3: sent TypeSafe the clip, your numbers and an offer of a case study or bug report
- [ ] Checkpoint 3: applied to agent and decision-model startups

### Phase 5 — Polish and visuals · 8–10 h · weeks 10–11

**Learn:** instanced 3D rendering in Taichi's GGUI or Three.js, lighting, colour by state, motion trails and camera paths (a slow orbit makes 3D read instantly).

**Build:** dark background, bots drawn as small dark-purple capsules (an original look, not the film's), a HUD showing bot count, FPS, the current command and Jev's confidence, plus a counter of local decisions per tick.

**Done when:** a 10-second clip looks good with no explanation.

**Post:** the polished reel on Instagram and a teaser on LinkedIn or X ("full write-up next week").

**Reach out:** no new cold contacts. Send a short preview to the 3–5 people who replied most warmly and ask for one piece of feedback before launch.

**Phase 5 checklist**

- [ ] 3D instanced rendering, colour by state, motion trails and an orbiting camera
- [ ] HUD with bot count, FPS, current command, Jev confidence and local decisions per tick
- [ ] 10-second clip that works with no explanation
- [ ] Instagram reel, plus a LinkedIn or X teaser with the launch date
- [ ] Ran a "which shape next?" poll and built the winner
- [ ] Before-and-after polish post
- [ ] Preview sent to the 3–5 warmest contacts for feedback

### Phase 6 — Package as proof of work · 6–8 h · weeks 11–12

Covered in its own section below.

**Post:** the launch, with the video, repo and write-up on the same day. Post on LinkedIn, X, Instagram, Show HN and r/MachineLearning (check its self-promotion rules first).

**Reach out (credibility checkpoint 4):** close the loop with everyone in your contact log, sharing the final result plus one line on what their reply changed. Then apply widely to co-op, research and internship roles with the project as the first line of your resume. Pitch a UWaterloo lab directly on extending the project with them.

**Phase 6 checklist**

- [ ] 45–60 s demo video with captions
- [ ] 10 s loop GIF
- [ ] Repo with a one-command run, README, architecture diagram, results and known limits
- [ ] Write-up of 800–1,200 words with metrics
- [ ] Launch thread on X, and posts on LinkedIn, Instagram, Show HN and r/MachineLearning
- [ ] Messaged the 5–10 most engaged followers with the launch
- [ ] Checkpoint 4: closed the loop with everyone in the contact log
- [ ] Checkpoint 4: applied widely and pitched a UWaterloo lab
- [ ] "What I learned building in public" thread a few days later

## Growing an X following alongside the build

Post short, visual progress almost every day and spend 20–30 minutes a day replying to people in the field. On X, a swarm clip and good replies to bigger accounts do more than any thread you write alone.

Rules that apply in every phase:

- **Video first.** Upload clips natively to X (not YouTube links), 5–15 seconds, with motion in the first second. A swarm moving is the hook.
- **One idea per post.** Lead with the result ("1,000 bots, no leader, forming a bridge"), then one line on how.
- **Cadence:** about 5 short posts a week plus 1 longer thread a week. Consistency beats quality spikes.
- **Reply strategy:** most early followers come from your replies, not your posts. Add real substance to posts by researchers and builders in ML, robotics and JAX.
- **Be around after posting.** Reply to every comment in the first 1–2 hours; it extends the post's reach.
- **Reuse everything:** each X clip doubles as an Instagram reel.
- **No shortcuts.** Skip bought followers and engagement pods; they hurt reach and credibility with exactly the people you want to reach.

Realistic expectations: most build-in-public accounts grow slowly for weeks, then jump when one clip hits. The self-healing clip is your best bet for that jump, so the earlier phases are about having an audience and a profile ready when it lands.

### Phase 0 — Set up the account

- Bio: one line saying who you are and what you're building, for example "First-year @UWaterloo CS/BBA · building self-healing microbot swarms in public."
- Pin a kickoff post that states the goal and the 12-week timeline, so new visitors know why to follow.
- Follow 100–150 accounts in swarm robotics, NCA, JAX, ML research and build-in-public. Turn on notifications for 10 of them so you can reply early.
- Post daily "today I learned" notes while learning JAX. Small and honest is fine.

### Phase 1 — First clips

- Post the first flocking clip with a hook about having no leader.
- Write your first mini-thread on one concrete lesson, such as how spatial hashing took the sim from N FPS to M FPS. Real numbers get shared.
- Quote-post relevant research or demos with your own angle instead of just liking them.

### Phase 2 — First real thread

- Post the first big thread: the 2014 Kilobot paper's result, then your simulated version, side by side, in 5–7 posts.
- Tag the authors only to credit them, never to ask for a reshare. Credit is what gets reshares anyway.
- Post on a weekday morning Eastern time, when North American tech X is most active.
- Update your pinned post to the best clip so far.

### Phase 3 — Go for the breakout

- Post the CNN-from-scratch mini-thread first; ML learners share "from scratch" content a lot.
- Treat the self-healing clip as your main event: a short clip (delete a chunk, watch it regrow), a one-line hook, and a follow-up thread ready to post as a reply.
- Post it when you can stay online for 2 hours to reply. Pin it.
- If it takes off, reply with a link to the repo or write-up so the attention has somewhere to go.

### Phase 3.5 — Show it adapting

- Post a side-by-side: the template NCA failing on a new layout next to the environment-aware one succeeding.
- Let followers suggest obstacle courses in the replies, run them, and reply with the clip. It's free content and gets people invested.
- Quote-post the Goal-Guided NCA paper with your result, crediting it.

### Phase 4 — Ride the Jev launch

- Jev is new, so people are searching for real examples. Post a clip of a typed command reshaping the swarm, with the confidence score visible on screen.
- Tag TypeSafe and show the accuracy numbers. Companies often reshare good outside demos of a new model.
- Write a short post on decision models versus LLMs for control, using your failure cases. Opinion with evidence tends to get the most engagement.

### Phase 5 — Bring followers into the build

- Run polls ("Which shape next?") and then build the winner. People follow builds they feel part of.
- Post before-and-after visuals of the polish work.
- Tease the launch with a date.

### Phase 6 — Launch day

- One launch thread: video, what it does, how it works, the repo, the write-up.
- Message the 5–10 people who engaged most along the way, share the launch, and let them decide whether to reshare.
- A few days later, post a "what I learned building this in public" thread with your follower and contact numbers. Meta-posts about the journey often outperform the project itself.

## Skills to learn

Tick these off as you go. The bold ones are the minimum for a working demo; the rest make it impressive.

**Programming and tooling**

- [ ] **Python with NumPy vectorization**: thinking in arrays, not loops
- [ ] **JAX core**: `jit`, `vmap`, `lax.scan`, pure functions, PRNG keys
- [ ] **Git and GitHub**: clean commits, a README, releases
- [ ] Flax (neural network layers) and Optax (optimizers)
- [ ] Async Python (`asyncio`) or threads, so API calls don't block the sim
- [ ] Google Colab with a free GPU for 3D training (WSL2 only if you want GPU JAX on your own laptop)

**Simulation and math**

- [ ] **Vectors: positions, velocities, distances, normalization**
- [ ] **Fixed-timestep simulation loops**
- [ ] **Spatial hashing or a cell list for neighbour search**
- [ ] Trilateration (position from distances to known points)
- [ ] Signed distance fields or bitmaps for "is this point inside the shape"

**Swarm robotics concepts**

- [ ] **Boids: separation, alignment, cohesion**
- [ ] **Decentralized control and emergence (local info only)**
- [ ] **Hop-count gradients and seed robots**
- [ ] Edge-following and finite state machines per agent
- [ ] Self-healing and robustness to agent failure

**Machine learning**

- [ ] Convolutions as local perception, in 2D and then 3D (voxel grids)
- [ ] Training loops: loss, gradients, learning-rate schedules
- [ ] Neural cellular automata: sample pool, damage augmentation, alive masks, and environment input channels
- [ ] Stretch: multi-agent RL basics, such as shared policies and reward shaping

**Decision layer**

- [ ] Calling a hosted model API with keys kept in environment variables
- [ ] Jev typed questions (Choice, Score) and confidence thresholds
- [ ] Prompt and state design: describing a cluster's situation in short text

**Presentation**

- [ ] Screen recording and trimming (OBS plus any editor)
- [ ] Writing a technical write-up: problem, approach, results, failures

## Resources

Read the ones marked "core" before or during their phase. The rest are for when you get stuck or want to go further.

| Resource | What it gives you | Phase | Priority |
| --- | --- | --- | --- |
| [JAX docs: quickstart and "Thinking in JAX"](https://docs.jax.dev/en/latest/) | `jit`, `vmap`, PRNG keys, the pure-function rules | 0 | Core |
| [Craig Reynolds: Boids](https://www.red3d.com/cwr/boids/) | The original three flocking rules, with diagrams | 1 | Core |
| [The Coding Train: Flocking simulation (YouTube)](https://www.youtube.com/watch?v=mhjuuHl6qHM) | Building Boids step by step, visually | 1 | Helpful |
| [Rubenstein et al., Programmable self-assembly in a thousand-robot swarm (Science, 2014)](https://www.science.org/doi/10.1126/science.1254295) | The gradient, localization and edge-following algorithm | 2 | Core |
| [Kilombo simulator (GitHub)](https://github.com/JIC-CSB/kilombo) | A reference Kilobot simulator to read code from | 2 | Helpful |
| [Mordvintsev et al., Growing Neural Cellular Automata (Distill, 2020)](https://distill.pub/2020/growing-ca/) | Self-healing NCA, with code and interactive demos | 3 | Core |
| [Flax docs](https://flax.readthedocs.io/) | Neural network layers in JAX | 3 | Core |
| [Optax docs](https://optax.readthedocs.io/) | Optimizers and learning-rate schedules | 3 | Helpful |
| [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev.md) | Model IDs, access without the waitlist, pricing | 4 | Core |
| [awesome-jev wiki (GitHub)](https://github.com/kraayenjon/awesome-jev/wiki) | Endpoint, SDK install, typed outputs, known failure modes | 4 | Core |
| [Beam AI: Jev for AI agents](https://beam.ai/agentic-insights/jev-typesafe-ai-agents) | Where a decision model fits in an agent architecture | 4 | Helpful |
| [Taichi Lang](https://www.taichi-lang.org/) | Fast GPU or CPU rendering and particle sims in Python | 5 | Helpful |
| [JaxMARL (GitHub)](https://github.com/FLAIROx/JaxMARL) | Multi-agent RL in JAX, if you go past the demo | Stretch | Optional |
| MIT M-Blocks (search "MIT M-Blocks self-assembling robots") | Real modular cubes, good inspiration for a 3D version | Stretch | Optional |

For the 3D half of Phase 3, read [Growing 3D Artefacts and Functional Machines with Neural Cellular Automata (Sudhakaran et al., 2021)](https://arxiv.org/abs/2103.08737) after the Distill article.

For Phase 3.5, read [Goal-Guided Neural Cellular Automata (Sudhakaran et al., 2022)](https://arxiv.org/abs/2205.06806) after the Distill article.

Two free courses fill gaps fast if needed: [3Blue1Brown's linear algebra series](https://www.3blue1brown.com/topics/linear-algebra) for vectors, and [fast.ai](https://course.fast.ai/) for training intuition.

## Tech stack and setup

The simulation and rendering run on an ordinary laptop CPU; the plan was sized for an HP ENVY with 32 GB of RAM, which is far more than needed. Training the 3D NCA is the exception: use a free Google Colab GPU, where it takes minutes instead of hours.

| Layer | Tool | Notes |
| --- | --- | --- |
| Language | Python 3.13 | In a virtual environment (`python -m venv .venv`) |
| Simulation core | JAX (CPU), NumPy | Native Windows JAX is CPU-only; GPU needs WSL2 and an NVIDIA card |
| Learning | Flax, Optax | Phase 3 onward; 3D training on a free Google Colab GPU |
| Rendering | Matplotlib for 2D, then a 3D viewer (Taichi GGUI or PyVista) | 3D needs a viewer with a movable camera; Matplotlib's 3D mode is too slow for thousands of bots |
| Decision layer | Jev via OpenRouter or Vercel AI Gateway | Direct TypeSafe access is waitlisted |
| Speech (optional) | Any speech-to-text API or local Whisper | Turns voice into the text command Jev reads |
| Recording | OBS Studio | 1080p, 60 FPS for smooth swarm motion |
| Repo | GitHub | Public from day one; commit each phase |

Setup checklist:

- [ ] Install Python and create `.venv`
- [ ] `pip install jax numpy matplotlib pillow`, and later `flax optax taichi`
- [ ] Put API keys in a `.env` file, and add `.env` to `.gitignore`
- [ ] Plug in and use the high-performance power mode while training; laptops throttle hard on battery
- [ ] Close the browser and other heavy apps during NCA training runs

Performance targets: about 1,000 bots at 30+ FPS for live demos. 2D NCA training on a 64×64 grid takes tens of minutes to a couple of hours on CPU. The 32×32×32 3D grid has about 8 times as many cells, so train it on Colab.

## Packaging as proof of work · 6–8 h

The video is what people watch; the repo and write-up are what convince technical readers. Ship all three together.

- [ ] **45–60 s demo video**: a cold open with the swarm forming a shape, then a voice or typed command, the morph, a chunk deleted, the self-heal, and a split screen of Kilobot vs NCA. Add captions, since most people watch muted.
- [ ] **10 s loop GIF** for the README header and social posts
- [ ] **GitHub repo**: one-command run (`python run.py --mode nca`), a README with the GIF, an architecture diagram, results and known limits
- [ ] **Write-up** of 800–1,200 words: what each bot knows, the three control approaches compared, what Jev decides and why, failures and fixes, and numbers such as bots, FPS, regrowth time and Jev accuracy on 20 test commands
- [ ] **Short posts**: a LinkedIn or X thread plus an Instagram reel, tagged #swarmrobotics and the JAX community
- [ ] **Outreach**: send the video and repo to people building in swarm robotics, multi-agent systems or decision models, including the TypeSafe team, with one specific question each

Metrics worth showing: bot count, FPS, time to form each shape, percentage of the shape recovered after 30% damage, and command accuracy with Jev versus keyword matching.

## Risks, scope cuts and stretch goals

The biggest risks are Phases 3 and 3.5 taking twice as long. If Phase 3 slips, ship the Kilobot version with Jev. If Phase 3.5 slips, ship with the template NCA and post the environment-aware version later as a follow-up. If 3D itself is the blocker, fall back to the 2D NCA from the first half of Phase 3, which is already a complete demo.

| Risk | Sign it's happening | Fallback |
| --- | --- | --- |
| NCA training won't converge | Loss is flat after an hour | Use a smaller grid (32×32) and a simpler shape, or copy the Distill reference setup exactly first |
| Laptop too slow | Below 20 FPS at 1,000 bots | Drop to 500 bots, render every second frame, or train 3D models on Colab |
| Kilobot edge-following jams | Bots pile up at the shape's edge | Add random jitter and a timeout state, or move to grid-snapped movement |
| Jev misreads commands | Wrong shape on clear commands | Make Choice labels more descriptive, raise the threshold, fall back to keywords |
| Jev access or latency issues | Calls fail or take seconds | Run calls async; the swarm keeps its current shape until an answer arrives |
| School crunch | Two weeks with no progress | Every phase ends with a postable result, so stop at any phase and ship that |

The minimum shippable demo is Phases 0–2 plus 4: a Kilobot-style swarm with Jev commands, about 35–40 hours.

Stretch goals, in order of payoff:

1. Bigger 3D swarms (64×64×64 grids or more) trained on a GPU
2. A physics close-up: 50–100 bots in PyBullet with magnetic-style links carrying a load
3. A learned per-bot policy with JaxMARL instead of the NCA grid
4. A browser version (Three.js plus a small backend) so anyone can try it from a link
