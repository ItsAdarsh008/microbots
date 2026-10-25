# How swarm self-repair emerges: a literature review for the new bot body

Question: how can shape self-repair *emerge* from local rules in a swarm of the microbots in `assets/microbot_v1`, given what the MuJoCo measurements say about the body
(`assets/microbot_v1/mujoco/FINDINGS.md`): each bot slides along ONE line perpendicular to its own fixed pole axis, with fine strides up to about 50 mm and a small drift along the
axis; it has docking connectors (genderless Hirth face + electro-permanent magnet) and, with the moves tried so far, cannot steer.

**How much I read.** *Read* = I read the abstract or full text on the source's own page. *Snippet* = only a search summary; treat as a lead, not a fact. I could not read the full text of several
older papers; I say so each time. "I found no paper on X" means my searches returned none, not that none exists.

## 1. What mechanisms the literature shows (and what each needs from the body)

| Mechanism | Source | What it does | Evidence level | Needs from the body |
| --- | --- | --- | --- | --- |
| **Density / "gas" model**: shape is a region, agents repel each other inside it, so after deaths the survivors spread to refill; internal holes collapse | SHAPEBUGS, Cheng, Cheng & Nagpal, AAAI 2005 | "adapts to influx and death of agents, as well as accidental displacement"; "recover from most patterns of agent death"; self-repair by "automatically collapsing internal holes" | *Read* (extracted PDF text: abstract and introduction). Simulation only. | Free 2D motion in discrete steps; **a compass** ("gives perfect directional orientation"); a shared coordinate system built by local trilateration; proximity sensing + radio to neighbours |
| **Density-field pattern control** | arXiv 2609.12959 (2026) | pattern as an implicit density field, so there is no agent-to-target assignment; "intrinsic self-healing and elastic reconfiguration"; flights with up to 15 quadrotors | *Read* abstract. No recovery numbers in the abstract (a search summary quoted 93.9% coverage vs about 83% at 50% loss; unverified). | Free 2D/3D motion (UAV dynamics), "model-agnostic" |
| **Self-maintaining primitives**: a gradient or a line that re-forms when processors fail; larger patterns built from self-repairing primitives inherit the repair | Clement & Nagpal, "Self-assembly and self-repairing topologies", 2003 | assumes units fail; uses an always-updating gradient | *Snippet* (project page and summary; full text not read) | Only local message passing; no motion needed for the gradient itself |
| **Gradient + edge-following assembly** | Rubenstein, Cornejo & Nagpal, Science 2014 (Kilobots) | 1,024 robots form 2D shapes; "highly robust to the variability and error"; about 12 h for a K shape | *Snippet*; I did not find self-repair or dead-robot results | Kilobots are slow, imprecise, nonholonomic (forward + turn) movers with ranging only |
| **Shape recovered at a smaller scale** | Rubenstein & Shen, IROS 2008 | "if robots are removed ... the remaining agents will recover the former shape, but on a smaller scale" | *Read* abstract only; mechanism not read | Not stated |
| **Stigmergy** | TERMES, Werfel, Petersen & Nagpal, Science 2014 | robots build by reading the structure; robust to losing a robot | *Snippet*; I found nothing on repair after removed blocks | Builders on the structure; not a free-moving swarm |
| **Neural cellular automata, regrowth** | Horibe, Walker & Risi, EvoStar 2021 (arXiv 2102.02579) | simulated soft robots regrow damaged parts from local cell information; regain "some" locomotion | *Read* abstract. Simulation only; no numbers on the page | Cells that can appear (voxels), not moving agents |
| **Neural cellular automata on hardware** | "Smart Cellular Bricks", arXiv 2509.18659 (2025) | about 200 physical bricks, same NCA on each, no global state; shape classification 98.97% (simulation), 100% in hardware on 4 shapes (consensus in about 23 iterations, 70 s); damage-direction detection above 90%; simulated recovery accuracy per class 0.64 to 0.95 | *Read* (full HTML). **It does not physically repair itself**: no actuation; regrowth is simulated by adding bricks from outside; the authors list active self-repair as future work. Tolerates 5 to 15% failed modules, except where a failure severs a narrow neck (guitar). | Perception and communication only: ESP32 per brick, serial links |
| **Lattice self-repair by moving modules** | Seremo / L-system reparation (HIT, 2019); Peck thesis (York, 2021); M-Cubes; DTU fault-tolerant reconfiguration (2007) | rebuild lost parts by directing other modules; swap or eject faulty modules; simulated 500-module robot with redundancy and roll-back | *Snippet* for all four | Docking and detaching on a lattice; reconfiguration planning |
| **Distributed formation estimators** | Image-moment formation control, arXiv 2312.07523 | "self-healing" estimator tolerant to packet drops and robots joining or leaving; 50 robots, about 50% packet loss | *Read* abstract. Says nothing on motion model or failures beyond that. | Free motion, each robot knows its own pose |
| **Learned multi-agent policies** | Pina, De Silva & Artaud, ICPRAM 2024 | MARL agents adapt to simple teammate faults but "do not adapt well" to complex ones | *Snippet* | n/a |

**Gap I found.** I did not find a paper that gets *shape* regeneration to emerge in a swarm of *moving* agents by *learned* local rules, on hardware or on a constrained body. The 2025 brick paper
is the closest hardware result and stops short of action; the NCA regrowth papers have cells that appear rather than agents that move. That is where this project sits, but my searches were limited.

## 2. What this implies for the new body

1. **Compass-based methods are out as written.** SHAPEBUGS assumes a perfect compass and free movement; our bots have neither. Its density idea, not its coordinate system, is what transfers.
2. **The density view fits our evaluation.** Our evaluator already rescales density to the surviving count, which matches "recover the former shape on a smaller scale" (Rubenstein & Shen 2008) and the gas-model equilibrium (a smaller swarm just lowers the density).
3. **One-line motion is the open question, not covered by anything I read.** Every moving-agent method above assumes motion in the plane. A swarm in which each agent can only slide along its own random fixed line is a different
   controllability problem; I found no paper on it. Our `ag_axis_*` runs test whether learned rules cope.
4. **Perception without action is cheap; action is the hard part.** Smart Cellular Bricks shows local NCA perception and consensus on commodity hardware. Combining that with a body that can actually move, even in one dimension, is the missing piece.
5. **Redundancy matters for connectivity.** The bricks' guitar failed at a narrow neck; our learned cells lose a fragment that is farther than the sensing radius from the body. Both say: the shape's thin parts are where repair breaks.
6. **Docking is an untapped resource.** The M-Blocks work (pivoting on magnetic hinges; *snippet*) shows modules moving by pivoting about neighbours. Our docked pairs could in principle move or turn each other, which a single bot cannot; I have not tested this.

## 3. Experiments this suggests (in order of cost)

1. Give the learned cell a **maintained gradient channel** (hop count from the damage boundary or from a seed), as in Clement & Nagpal; compare with the memory-channel cells. Cheap; in the abstraction.
2. **Density-deficit input** to the cell (it already gets a density gradient) plus a loss on density only; test whether geometry repair improves.
3. **One-line motion**: `ag_axis_*` runs (queued) with the gradient channel added.
4. **Steering by docking** in MuJoCo: two or three bots, a weld constraint standing in for the EPM, measure whether a docked pair can turn or translate in any direction.
5. Only after that: **particle division** (spare daughter slots) for cuts.

## Sources
- Cheng, Cheng, Nagpal. Robust and Self-Repairing Formation Control for Swarms of Mobile Agents. AAAI 2005. https://jmvidal.cse.sc.edu/library/cheng05a.pdf
- Smart Cellular Bricks for Decentralized Shape Classification and Damage Recovery. https://arxiv.org/html/2509.18659v2
- Horibe, Walker, Risi. Regenerating Soft Robots through Neural Cellular Automata. https://arxiv.org/abs/2102.02579
- Distributed Stochastic Optimal Control for Pattern-Oriented Swarms. https://arxiv.org/abs/2609.12959
- Self-Healing Distributed Swarm Formation Control Using Image Moments. https://arxiv.org/abs/2312.07523
- Rubenstein, Shen. A Scalable and Distributed Approach for Self-Assembly and Self-Healing of a Differentiated Shape. IROS 2008. https://robots.isi.edu/prl/b2hd-rubenstein2008-approach-for-self-assembly-and-self-healing-of-a-differentiated-shape.html
- Rubenstein, Cornejo, Nagpal. Programmable Self-assembly in a Thousand-Robot Swarm. Science 2014. https://ssr.princeton.edu/publications/programmable-self-assembly-thousand-robot-swarm
- Werfel, Petersen, Nagpal. Designing collective behavior in a termite-inspired robot construction team. Science 2014. https://ssr.princeton.edu/document/691
- Clement, Nagpal. Self-assembly and self-repairing topologies. 2003. https://groups.csail.mit.edu/mac/projects/amorphous/paperlisting.html
- Romanishin, Gilpin, Claici, Rus. 3D M-Blocks. https://people.csail.mit.edu/sclaici/publication/mblocks
- Peck. Dynamic Self-repair (PhD thesis, York). https://etheses.whiterose.ac.uk/30288/
