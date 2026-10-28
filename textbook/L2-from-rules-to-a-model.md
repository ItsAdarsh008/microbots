# Linking 2: From rules to a trained model

**Side track · Read before:** talking through the learned version with a collaborator · **Code:** `linking/learn/` (being built alongside this chapter)

In Linking 1 the swarm builds bridges, domes and letters, but every decision comes from rules someone wrote by hand. That is the first thing a machine-learning person will ask about: *what architecture, what policy?* Right now the honest answer is "none, it's rules." This chapter explains what it takes to turn those rules into a **trained model**, using the same ideas as Neural Particle Automata (NPA), so you can talk about it clearly.

The first version is built: see Part 7 for what happened, and `linking/learn/` for the code.

---

## Part 1: Rules versus a model

### 1.1 What the sandbox does now

Every step, each free bot:

1. looks up a **distance map**: how many moves from each cell to the nearest empty target cell;
2. lists its **legal moves** (slides and pivots);
3. picks a move that lowers its distance;
4. keeps the move only if nothing comes loose.

Steps 1 and 3 are the "brain," and both were written by hand. Worse, the distance map is computed over the **whole grid**, which no real bot could see.

### 1.2 What "a model" would mean

Replace step 3 (and eventually step 1) with a small **neural network** that every bot runs:

- **Input:** what this bot can see around itself, as a list of numbers.
- **Output:** a score for each move it could make.
- The bot picks the highest-scoring legal move.

Nobody writes the scores by hand. The network's numbers (its **parameters**, or **weights**) are adjusted by **training** until the bots build well.

Three words you'll hear constantly:

| Word | Meaning here |
| --- | --- |
| **Observation** | The input: what one bot sees this step |
| **Action** | The output: which move it makes |
| **Policy** | The rule that turns an observation into an action. "Learned policy" means the network *is* the rule. |

### 1.3 One network, every bot

All bots run **the same network** with the same weights. Each one just feeds it a different observation, because each one stands somewhere different. This is called **parameter sharing**, and it's exactly how NPA and the Growing NCA work: one tiny program copied into every cell or particle. It means:

- the network stays small (thousands of numbers, not billions);
- it works for any number of bots, since adding a bot doesn't change the network;
- every decision is **local**: a bot only uses what's near it.

---

## Part 2: What Kuan's NPA does

Neural Particle Automata (NPA) is the method Kuan has been training (the regrowing lizard). In outline:

- Each **particle** has a position and a small list of hidden numbers (its **state**).
- Each step, a particle **perceives** its neighbours: it blends their states, weighted by how close they are.
- One **shared MLP** (a small network, explained below) takes that perception and outputs a change: how to move and how to update its state.
- Particles update at random times rather than all at once, which makes the result robust.
- Training adjusts the MLP until the particles form the target shape and regrow it after damage.

Two differences from our sandbox matter:

| | NPA | Our sandbox |
| --- | --- | --- |
| Where bots can be | anywhere (continuous space) | grid cells only |
| What a move is | a small push in any direction | one of a few legal slides or pivots |
| Staying attached | not modelled | the hard rule everything depends on |

So we can borrow NPA's structure (shared network, local perception) but our **output** has to be "which legal move," not "push this way." That is the main thing to agree on with Kuan.

### 2.1 What an MLP is

A **multi-layer perceptron (MLP)** is the simplest neural network. It is just a few rounds of: multiply the inputs by a table of weights, add a bias, and squash out the negatives (a **ReLU**). For example:

```
258 numbers in  →  [weights 258×128] → ReLU  →  [weights 128×128] → ReLU  →  [weights 128×19]  →  19 scores out
```

That's about 52,000 weights in total. Training is the process of nudging those 52,000 numbers.

---

## Part 3: "Input and output format"

When people say the **format**, they mean: exactly which numbers go in, in what order, and exactly what the numbers coming out mean. Two pieces of code can only work together if they agree on it. It can always be changed later with a small translation layer, but agreeing early saves work.

Here is the format this project will start with. Every number below is a starting choice, and all of it can change after talking to Kuan.

### 3.1 Observation: 258 numbers per bot

| Part | Count | What it is |
| --- | --- | --- |
| Neighbourhood: occupied? | 125 | A 5×5×5 block centred on the bot: 1 if a bot is there, 0 if empty (cells outside the world count as empty; the floor row counts as occupied) |
| Neighbourhood: part of the shape? | 125 | Same block: 1 if that cell belongs to the target shape. Like the Kilobots, every bot is given the shape map. |
| Work signal | 7 | The "how far to the work" value at the bot's own cell and its 6 face neighbours (Part 4 explains how it becomes local) |
| Own state | 1 | 1 if the bot is locked into the shape, 0 if it's free |

### 3.2 Action: 19 slots

Every move a bot could ever make lands it in one of the 18 cells around it (6 sharing a face, 12 sharing an edge), or it stays put:

- slot 0: **stay**
- slots 1 to 6: the 6 face cells (slides)
- slots 7 to 18: the 12 edge cells (pivots)

The network outputs 19 scores. Before choosing, every slot that isn't a legal move right now is **masked**: its score is set to minus infinity, so it can never be picked. Moves that would leave anything unattached are masked too. So the network decides *which* safe move, and the safety rules still decide *what's* safe. This is called an **action mask**.

### 3.3 Reward (for the second stage of training)

| Event | Reward |
| --- | --- |
| A target cell gets filled | +1 |
| Every step that passes | −0.01 (so faster is better) |
| The shape is finished | +10 |

There is no penalty for breaking the structure, because the mask makes breaking it impossible.

---

## Part 4: Making the "work signal" local

The distance map is computed over the whole grid. A real bot can't do that. The fix is the **hop-count gradient** from Chapter 2 (the Kilobots):

- Bots on the edge of the unfinished shape broadcast 0.
- Every other bot listens to its neighbours and broadcasts one more than the smallest value it hears.
- After a few rounds, every bot holds roughly how many hops it is from the work, using only messages from neighbours.

That number is the "work signal" in the observation. It's not identical to the global distance map (it travels through bots, not empty air), so the trained policy has to cope with a noisier signal. That's part of what training is for.

The "will anything fall?" check stays global for now. Making *that* local is a separate, harder problem, and it's honest to say so.

---

## Part 5: How it gets trained

### 5.1 Stage 1: imitation (copy the rules)

1. Run the hand-written rules on thousands of builds.
2. At every step, record each bot's observation and the move the rules picked.
3. Train the network to predict the rules' move from the observation. This is ordinary classification: the loss is **cross-entropy**, the same as image classification.

Result: a network that behaves like the rules, but **using only local information**. That alone is a real step, because the rules had a global map and the network doesn't.

This is called **imitation learning** or **behaviour cloning**. It's fast (minutes to hours on a free Colab GPU) and almost always works.

### 5.2 Stage 2: reinforcement learning (beat the rules)

Start from the imitation network and let it try things, rewarding it with the table in 3.3. The usual algorithm is **PPO** (Proximal Policy Optimization). Over many builds it nudges the weights toward moves that led to more reward.

This is where something new could appear: for example, with the real-strength limit on, does it learn to put support pillars down *before* the deck needs them, with nobody telling it what a pillar is? That's the interesting experiment, and it may take many tries or not work.

### 5.3 How we'll know it worked

| Check | Rules today | Target for the model |
| --- | --- | --- |
| Shapes finished | all 8 | all 8 |
| Ever disconnected | never | never (guaranteed by the mask) |
| Steps to build the long bridge | ~170 to 225 | same or fewer |
| Long bridge with real strength: support bots needed | 245 (planned by hand) | fewer, without a planner |

---

## Part 6: The plan

| # | Step | Time |
| --- | --- | --- |
| 1 | Port the simulation to Python/JAX, so thousands of builds can run in parallel | about a day |
| 2 | Build the observation, action and mask from Part 3, and write them up | half a day |
| 3 | Swap the global distance map for the local hop-count signal (Part 4) | half a day |
| 4 | Imitation training (done on a laptop CPU, ~6 minutes; see Part 7); next, load it into the sandbox with a "Rules / Learned" switch | done for the Python simulation |
| 5 | Reinforcement learning on the long bridge with real strength | 1 to 3 weeks, uncertain |

---

## Part 7: What happened when it was trained

The plan above was carried out on the Python version of the simulation (wall, arch and tower), with one change: the action-slot network shown in Part 2.1 (52,115 weights).

1. **Imitation alone** built the wall every time but finished the arch 2 times out of 5 and the unseen tower 3 times out of 5. The network agreed with the teacher on about 63% of decisions; the teacher is partly random, so that's close to the best possible.
2. **DAgger** (let the student drive, have the teacher say what it would have done, retrain) made the wall as fast as the teacher but didn't fix the others.
3. **A diagnostic**: giving the network the teacher's global map made it build everything, 15 out of 15. So the network wasn't too small. The problem was the work signal: passed only between touching bots, it takes long winding routes through the pile and points bots the wrong way.
4. **The fix**: relay the signal between bots up to 2 cells apart, like the Kilobots' infrared. Still local. Result: **15 out of 15**, including the tower it had never seen.

The lesson worth remembering: when a learned policy fails, test whether it's missing *information* or missing *capacity* before making the network bigger. Here one diagnostic run answered that.

---

## Questions to ask Kuan

1. Does your NPA code take one list of numbers per particle and output one action per particle? Could the output be "pick one of 19 moves" instead of a direction?
2. What does a particle see in yours: just neighbours within some radius? How big a radius?
3. How do you train it: copying a target, or trial and error? What's your loss or reward?
4. Do you keep hidden state per particle between steps? (Ours has none yet. Adding some could help bots remember what they were doing.)
5. Would you rather we plug the bridge into your training code, or keep two pipelines and compare?

---

## Check your understanding

<details>
<summary>1. What is the difference between the sandbox's rules and a learned policy?</summary>

The rules were written by hand and use a map of the whole grid. A learned policy is a neural network whose weights were found by training, and it only sees what's near each bot.
</details>

<details>
<summary>2. Why does every bot use the same network?</summary>

Parameter sharing: one small network copied into every bot, like NPA and NCA. It keeps the model small, works for any number of bots, and keeps every decision local.
</details>

<details>
<summary>3. What does "input and output format" mean, and can it change later?</summary>

Exactly which numbers go in (the observation) and what the numbers coming out mean (the action). It can change later with a small translation layer; agreeing early just saves work.
</details>

<details>
<summary>4. Why can't the network break the structure?</summary>

Action masking: illegal moves and moves that would leave anything unattached get a score of minus infinity before the bot chooses, so they can never be picked.
</details>

<details>
<summary>5. What is imitation learning, and why do it first?</summary>

Training the network to copy the rules' choices from local observations. It's fast, reliable, and gives a working local policy to start reinforcement learning from.
</details>

<details>
<summary>6. What part is still not local, even after this plan?</summary>

The "will anything fall?" check. It still looks at the whole structure.
</details>

---

## Glossary

| Term | Meaning |
| --- | --- |
| Model | A neural network with trained weights |
| Weights / parameters | The numbers inside a network that training adjusts |
| Observation | What one bot sees in one step, as a list of numbers |
| Action | What a bot decides to do |
| Policy | The mapping from observation to action |
| MLP | Multi-layer perceptron: layers of multiply, add, ReLU |
| ReLU | Replace negative numbers with 0 |
| Parameter sharing | Every bot runs the same network |
| Action mask | Blocking illegal choices before the bot picks |
| Imitation learning / behaviour cloning | Training a network to copy an existing set of decisions |
| Cross-entropy | The usual loss for "pick one of these options" |
| Reinforcement learning (RL) | Training by trial and error with rewards |
| PPO | A common, stable RL algorithm |
| Hop-count gradient | A distance passed bot to bot, one more than the smallest neighbour's value |
| NPA | Neural Particle Automata: one shared network in every particle, perceiving its neighbours |
