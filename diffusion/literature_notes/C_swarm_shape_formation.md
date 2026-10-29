# C. Swarm-specific and shape-formation literature

Compiled 2026-10-29. Tags: [verified] = I fetched the arXiv/journal page and read the abstract/page text; [snippet only] = only a search excerpt or secondary write-up; [from memory] = neither. "Verified" means the ABSTRACT page, not the full paper, unless stated. Numbers not in the abstract are marked as unknown rather than guessed.

## 1. Bottom line

1. I found NO paper that uses diffusion / flow matching to make a swarm of locally-sensing agents self-assemble into a target shape. Diffusion in multi-agent work is used for trajectories, policies, and global-state reconstruction, almost always with a central planner or global attention. The idea is open ground, which is good for novelty and bad for baselines.
2. The strongest non-learned competitor is the mean-shift family (Sun et al., Nat. Commun. 2023): shape broadcast as a distance-transform grid to every robot, 50 real robots, 16 to 1,024 simulated, self-repair, no goal assignment. Any learned method must be compared against something this simple.
3. The only learned decentralized assembly results that generalise to unseen targets (Blocks Assemble, GNN/attention + RL) needed ~1e9 env steps and structured (graph) observations; no paper I found quantifies "how many training shapes", so your 10k-shape sweep with held-out shapes would itself be a result.
4. Hierarchy is the norm: a global description (density, grid, SDF, sample points, tree) is broadcast or stored per robot, and local rules read it. Your noise-to-shape diffusion is a different hierarchy: the "shape code" is a conditioning vector and the denoiser must be local. Kamb and Ganguli (2412.20292) give theory for why locality plus equivariance still generalises in image diffusion.
5. Biggest physical-realisability risks: a shared global clock/noise schedule `t`, global attention over all N particles, particle identity/ordering, unbounded per-step displacement, no collisions, and fixed N. Mean-field/density views (Eulerian control, Wasserstein flows) are the cleanest answer to identity and variable-N.

## 2. Per-paper entries

### 2.1 Learned decentralized shape formation / self-assembly

**Neural Particle Automata: Learning Self-Organizing Particle Dynamics** [verified]
Kim, Pajouheshgar, Susstrunk, Jakob, Park. arXiv 2601.16096 (v1 22 Jan 2026, v3 24 Jun 2026), cs.NE.
Lagrangian generalisation of NCA: each particle has position + internal state and shares one learned rule, with differentiable SPH operators and CUDA kernels. Tested on morphogenesis, point-cloud classification, texture synthesis; claims NCA-like robustness and self-regeneration. Abstract says naive local interaction scales quadratically with N; no numbers given on the page.
Lesson: this is your prior baseline (single shape per model). The comparison "one model, 10k shapes, conditioned" vs "one NPA per shape" is the headline.

**Blocks Assemble! Learning to Assemble with Large-Scale Structured Reinforcement Learning** [verified abstract; block range from search snippet only]
Ghasemipour, Freeman, David, Gu, Kataoka, Mordatch. arXiv 2203.13733 (ICML 2022).
Agents with graph-structured observations and attention, trained with RL, "generalize to complex unseen blueprints in a zero-shot manner" (abstract). Snippet says training blueprints had 2 to 16 blocks and about 1e9 environment steps; I did not confirm in the body.
Lesson: zero-shot generalisation to unseen targets is achievable, but it is a CENTRAL controller with global graph observation, not local sensing. Evidence that a diverse blueprint distribution matters; cost of RL is huge compared with supervised/generative training.

**Reconfiguration of pivoting cube ensembles under local sensing constraints using geometric deep learning** [verified]
Dobreva, Blazquez, Grover, Izzo, Qin, Dold. arXiv 2509.03140 (v1 3 Sep 2025; published Astrodynamics 2026).
2D pivoting-cube modular robots; each cube has a small network using only nearby cubes; cube selection is globally coordinated. Even the most local variants reach the target; performance improves with more information passing; near-optimal with nearest-neighbour interactions plus multiple message-passing rounds. Grid symmetries (rotation/mirror) gave only minor training benefit but allowed smaller models. Abstract mentions a single target shape and gives no cube counts.
Lesson: multiple rounds of neighbour message passing recover most of global information. Your denoiser could use K local rounds per step. Note the "globally coordinated cube selection" caveat (same flavour as the pitfalls below).

**Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks** [verified]
Transue, Wang. arXiv 2502.17612 (Feb 2025), cs.RO/cs.LG.
GNN flocking controllers with built-in rotation equivariance and translation invariance reach similar performance with "70% less training data and 75% fewer trainable weights" and generalise better. Task is flocking, not shape formation.
Lesson: bake SE(2) equivariance into the local denoiser; expect a data-efficiency gain of that order. Shape CONDITIONING breaks global rotation symmetry, so decide whether the target is in a shared frame.

**Learning Decentralized Flocking Controllers with Spatio-Temporal GNN** [snippet only]
Virginia Tech group, ICRA 2024, arXiv 2309.17437 (authors not read). Imitation of a centralized expert; GNN fuses delayed states from distant neighbours with previous states of near ones.
Lesson: communication delay is modelled explicitly; your step-synchronous denoiser should be tested with delayed or stale neighbour info.

**Omotuyi and Kumar, GNN imitation for heterogeneous segregation/aggregation (J. Dyn. Syst. Meas. Control, 2024)** [snippet only, second-hand]
Reported via a related-work paragraph: local-sensing GNN trained by imitation of a centralized expert, comparable to the expert, scales to 100 robots in 2D and 3D, transferred to Turtlebot3 and Crazyflie. I did not see the original; title and details unconfirmed.
Lesson: "imitate a global teacher with a local student" is an established pattern; it is exactly your teammate's MLP baseline.

**Collisionless Pattern Discovery in Robot Swarms Using Deep Reinforcement Learning** [verified abstract]
Sharma, Ghosh, Misra, Mukhopadhyay, Sharma. arXiv 2209.09865 (Sep 2022).
DRL with constraint-shaped reward (mutual visibility, safe proximity) finds gathering and visibility patterns for fat robots from any start. No numbers in abstract.
Lesson: collision and visibility constraints can be put in the reward; for a generative model they must become projection or guidance steps.

**Boroson et al., distributed policy-gradient shape formation under partial observability (IROS 2017)** [snippet only; exact title not confirmed]
Search excerpt (act.usc.edu/publications/Boroson_IROS2017.pdf): model-free policy gradients, agents observe nearby agents but cannot identify or communicate; four agents forming a square; training time scales linearly with group size. I do not have the title or author list; cite with care.
Lesson: anonymous neighbours (no identity) was already accepted as the setting.

**Goal-Guided Neural Cellular Automata (GoalNCA)** [snippet only]
Sudhakaran, Najarro, Risi, 2022 (arXiv 2205.06806 per search URL, ICLR 2022 listing seen). Goal encoding applied at every step controls cell behaviour; reportedly keeps performance when only some cells receive the goal.
Lesson: direct precedent for "shape code as conditioning of a local rule"; partial broadcast to a fraction of agents is a testable axis for you. Number of training goals vs generalisation not found.

**Growing Neural Cellular Automata** (Mordvintsev et al., Distill 2020) [from memory]
Single-target NCA with regeneration; the origin of the NCA line NPA builds on. Cite only as background.

**Programmable self-assembly in a thousand-robot swarm** [snippet only]
Rubenstein, Cornejo, Nagpal. Science 345(6198):795-799, 2014.
1,024 Kilobots (about 3 cm, around $14 each), identical program, IR communication within about 3 body diameters, shapes such as a star, the letter K, a sea star; correction rules for traffic jams and drift; one IR command starts the run. Limitations noted: poor straight-line motion, noisy per-robot distance sensing.
Lesson: the physical existence proof for your Stage 2 constraints (noisy motion, range-limited comms, global start signal only). The target is a fixed image stored on every robot, not learned.

**Shape formation in homogeneous swarms using local task swapping** [snippet only]
Wang, Rubenstein, IEEE Trans. Robotics (2020; volume/issue not confirmed). Distributed algorithm solving goal assignment and collision-free routing together; up to 1,024 simulated and 100 real robots; news coverage says a set shape in under a minute.
Lesson: baseline for "time to form" and for identity-free assignment; shows assignment is the hard part that a density view avoids.

**Mean-shift exploration in shape assembly of robot swarms** [verified via PMC full text excerpt]
Sun, Zhou, Ma, Li, Gross, Chen, Zhao. Nature Communications 14:3476, 2023.
Target is a black-and-white image, converted to a binary grid then a distance-transform grayscale grid, sent to every robot and stored locally (tens of kB; cell count need not equal robot count). Robots use neighbour positions/velocities in a sensing radius and nearby grid gray levels; a robot surrounded by neighbours moves to the densest nearby pocket of free target cells; no goal assignment. Starfish: 16 to 1,024 robots (512 mismatched-count sims), coverage above 93%, entering rate 100%; snowflake 100% coverage but 75% without the shape-exploring term; at 300 robots convergence at least 20x faster than baselines; hardware with 50 robots; "shape regeneration" after removing robots. Used a mocap system and a shared global frame.
Lesson: the best reference baseline and the best source of metrics (see section 2.5). Variable robot/cell ratio and self-repair come for free from the density view. Its per-robot target map is the thing your diffusion approach must beat or justify replacing.

**Memory-Efficient 2D/3D Shape Assembly of Robot Swarms** [verified]
Yue, Li, Xu, Ze, Long, Cao, Sun. arXiv 2509.26518 (v1 30 Sep 2025, v2 23 Feb 2026).
Tree-based hierarchical encoding of 2D/3D target shapes, behaviour-based distributed assignment-free controller. 10 to 100x lower memory and 2 to 4x faster shape entry than the mean-shift algorithm (in simulation); hardware 6 to 7 UAVs. No learning.
Lesson: how the target is encoded per robot is itself a research axis; an octree/quadtree of the shape is a candidate conditioning format that is small enough to broadcast.

**Meanshift Shape Formation Control Using Discrete Mass Distribution** [verified abstract]
Cai, Gao, Li, Wang, Sun, Lu. arXiv 2602.00980 (1 Feb 2026).
Represents the shape as weighted sample points instead of a continuous density; claimed to handle swarm-size changes. No numbers in abstract.
Lesson: a sample-point target is the same data type as your particle set, so Chamfer/Wasserstein between swarm and target is natural.

**Huddle: Parallel Shape Assembly using Decentralized, Minimalistic Robots** [verified]
Chin, Meng, Chen, Bassett, Ivanov. arXiv 2603.17768 (18 Mar 2026).
Arbitrary-shape assembly by simple robots; assembled robots signal and align passing robots; neighbour-only communication; asynchronous; proof of no gaps / unreachable states; demonstrated with 107 robots.
Lesson: the asynchronous, no-pose, no-global-clock regime is feasible with hand design. A generative sampler with a global step counter is not in this regime.

**Distributed Stochastic Optimal Control for Pattern-Oriented Swarms** [verified abstract]
Zhang, Yu, Xue, Wang. arXiv 2609.12959 (Sep 2026).
Geometric patterns as implicit density fields (SDF-style implicit target), Bayesian-inference framing, supports self-healing and elastic reconfiguration; 15 quadrotors indoors, 4 outdoors.
Lesson: recent evidence that implicit-field targets are the mainstream way to describe a pattern to a swarm.

**Multi-Modal Decentralized RL for Modular Reconfigurable Lunar Robots** (arXiv 2510.20347) [snippet only]
Per snippet, modules learn their own policies and claim transfer to new morphologies; task is locomotion/manipulation, not shape assembly. Low relevance.

**SMORES-EP, a Modular Robot with Parallel Self-assembly** (arXiv 2104.00800; Autonomous Robots 47, 2023) [snippet only]
Target as a graph, module-to-vertex assignment by optimisation, concurrent docking. Non-learned. M-Blocks and Kilobots-era programmable-matter papers beyond the above were not searched in depth.

### 2.2 Diffusion / flow matching for multi-agent coordination

**MADiff: Offline Multi-agent Learning with Diffusion Models** [verified]
Zhu, Liu, Mao, Kang, Xu, Yu, Ermon, Zhang. arXiv 2305.17330 (v5 Jan 2025), NeurIPS 2024.
Attention-based diffusion network over all agents' trajectories; works as a decentralized policy (with teammate modelling) or a centralized controller; outperforms offline MARL baselines. No numbers in abstract.
Lesson: the attention is across all agents, i.e. global; for decentralized execution it models teammates, not neighbours within a radius. Not a shape generator.

**MIMIC-D: Multi-modal Imitation for Multi-agent Coordination with Decentralized Diffusion Policies** [verified]
Dong, Bhatt, Choi, Mehr. arXiv 2509.14159 (v1 Sep 2025, v4 Sep 2026).
Joint training with decentralized execution; each agent diffusion policy acts on local information; better multi-modal coordination in simulation and hardware. No numbers in abstract.
Lesson: closest in spirit to "local diffusion policy"; but small teams. Shows that diffusion keeps modes where regression averages them.

**CoDiMAD: Diffusion-Based Privileged Distillation for Communication-Free Multi-Robot Coordination** [verified]
Tao, Xin, Shen, Zhao, Zhang. arXiv 2607.09587 (10 Jul 2026).
MAPPO global-state teacher distilled into per-agent conditional diffusion students on local observation; argues regression distillation mode-averages. Three tasks; no numbers in abstract.
Lesson: directly relevant to your comparison with the teammate's MLP-from-teacher approach: when the same local observation is consistent with several global shapes or symmetric choices, the MLP will average. Design a test with symmetric targets to expose this.

**On Diffusion Models for Multi-Agent Partial Observability: Shared Attractors, Error Bounds, and Composite Flow** [verified]
Wang, Dong, Jiang, Parkes, Tambe. arXiv 2410.13953 (Oct 2024, v3 Feb 2025).
Diffusion models reconstruct global state from local action-observation histories; conditioned models settle on fixed points; composite agent-by-agent diffusion with convergence guarantee.
Lesson: theory that local conditioning can yield shared attractors representing the global state. Agent-by-agent updates are sequential, which matters for asynchronous swarms.

**Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies (OMAD)** [verified]
Li, Zhong, Wang, Xia, Zhang, Huang. arXiv 2602.18291 (Feb 2026, v2 Jun 2026).
Online off-policy MARL with diffusion policies, CTDE; SOTA on 10 MPE/MAMuJoCo tasks with 2.5x to 5x sample-efficiency gain.
Lesson: generic MARL; low direct relevance except as the "RL route" comparator.

**SwarmDiff: Swarm Robotic Trajectory Planning in Cluttered Environments via Diffusion Transformer** [verified abstract]
Ding, Jiao, Hu, Zhou, Wu, Mu, Liu. arXiv 2505.15679 (21 May 2025; CVPR 2025 workshop per search).
Hierarchical: the swarm is a probability density; a conditional diffusion transformer generates risk-aware (Wasserstein + CVaR) macroscopic trajectory distributions; per-robot MPC tracks them. Abstract has no numbers; a third-party review snippet (pith.science) claimed about 1 s planning for up to 500 robots and flagged an error in the CVaR equation, so treat safety claims as unconfirmed.
Lesson: the closest existing "diffusion for a swarm" design is a global-density planner plus local trackers. This is the hierarchy you would fall back to if a fully local denoiser fails.

**Simulation-Informed Diffusion for Decentralized Multi-robot Motion Planning (SID)** [verified]
Liang, Koenig, Fioretto. arXiv 2605.27697 (26 May 2026).
Each robot runs a constraint-aware diffusion model to forecast neighbours and plan its own path from local sensing; communication only in congestion; scales to 108 robots and 160 obstacles.
Lesson: shows decentralized diffusion with local sensing is possible at about 100 agents for navigation. Not shape formation.

**Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization** [verified]
Idoko, Jadhav, Singh. arXiv 2510.09204 (Oct 2025, v4 Jun 2026).
DiT flow-matching sampler with permutation-invariant robot and map encoders, plus a learned safety filter; tens of robots in tens of ms; claims orders of magnitude faster than diffusion baselines.
Lesson: flow matching plus a constraint-projection layer is the practical recipe for collisions; also permutation invariance by design. Centralized.

### 2.3 Denoising / Langevin dynamics vs collective dynamics, mean-field and PDE views

**Unifying GANs and Score-Based Diffusion as Generative Particle Models** [verified]
Franceschi, Gartrell, Dos Santos, Issenhuth, de Bezenac, Chen, Rakotomamonjy. arXiv 2305.16150, NeurIPS 2023.
Treats score-based diffusion and Wasserstein gradient flows as particle models where each sample's update depends on the population's current distribution.
Lesson: the formal bridge between "particles driven by a drift that depends on the population" (swarm) and a generative model. The learned score is the interaction rule.

**Gradient Flows as Optimal Controlled Evolutions: From R^n to Wasserstein product spaces** [verified abstract]
Chen, Georgiou, Pavon. arXiv 2510.27120 (31 Oct 2025), math.OC.
Casts Wasserstein gradient flow of relative entropy as optimal controlled evolution (Schrodinger-bridge motivation) and suggests microrobotic swarm control as an application.
Lesson: theory-only suggestion; supports the idea that noise-to-shape with a known endpoint is a Schrodinger-bridge / stochastic-control problem.

**Density control of large-scale particles swarm through PDE-constrained optimization** [verified]
Sinigaglia, Manzoni, Braghin. arXiv 2104.06373 (Apr 2021, v2 Jan 2022), submitted to IEEE T-RO.
Kolmogorov forward equation for density, boundary/global actuation, adjoint-based optimal control; simulations bring density close to target.
Lesson: global-actuation, underactuated swarms (cf. magnetically driven microbots) can be steered at the density level without addressing individuals.

**Target-density formation in swarms with stochastic sensing and dynamics** [verified abstract]
Hindes, Stantchev, Szwaykowska Kasraie, Schwartz. arXiv 2507.20911; Phys. Rev. Research 7, 043167 (2025).
Agents random-walk on a networked domain using noisy local density measurements; mean-field and finite-N analysis of how close and how fast the target density is reached vs N, sensing noise, collisions.
Lesson: gives scaling relations to compare with your measured error vs N.

**Probabilistic swarm guidance using optimal transport; Homogeneous Markov chain (HMC) shape formation; Distributed optimal transport for deployment** [snippet only]
Search summaries of Markov-chain bin-based shape formation (stationary distribution = target), PSG-OT, and a 2018 CDC paper on Wasserstein gradient-flow deployment. Authors/venues not read (arXiv 1403.4134 "Probabilistic and Distributed Control of a Large-Scale Swarm of Autonomous Agents" appeared; not fetched).
Lesson: bin-based methods need many bins for fine detail (resolution bottleneck); OT methods trade transitions for speed. Good classical baselines for the Eulerian view.

**An analytic theory of creativity in convolutional diffusion models** [verified]
Kamb, Ganguli. arXiv 2412.20292 (Dec 2024, ICML 2025 oral).
Locality (finite receptive field) and translation equivariance explain how diffusion models create novel samples as patch mosaics of training data; an analytic local-score machine predicts trained CNN diffusion outputs with median r^2 about 0.94 to 0.96 (CIFAR10, FashionMNIST, MNIST, CelebA); attention helps global coherence (r^2 about 0.77 on a self-attention UNet).
Lesson: a local denoiser is exactly a "local score machine"; predicts that a swarm denoiser with radius R produces shapes that are locally consistent mosaics of training-shape patches and fails on global coherence (e.g. wrong number of arms). Strong, testable hypothesis for your Stage 1. (Overlaps topic B; included because it is the one theory link between locality and generalisation.)

Searched but nothing found: any paper linking active-matter / Brownian-particle models explicitly to score-based generative modelling for robots (search 2026-10-29). Active-matter hits were physics papers on swarming thresholds (not read).

### 2.4 Global-to-local hierarchies and how the target reaches the swarm

Summary of what is actually done, from the entries above:
- Per-robot stored grid / distance-transform map: Sun et al. 2023 (tens of kB) [verified].
- Tree/quadtree-octree code: Yue et al. 2509.26518 [verified].
- Weighted sample points: Cai et al. 2602.00980 [verified abstract].
- Implicit density/SDF-like field: Zhang et al. 2609.12959 [verified abstract].
- Single broadcast start command, shape pre-programmed: Kilobots [snippet only].
- Global density transport planner then per-robot commands: Eulerian methods [snippet only]; SwarmDiff [verified abstract].
- Goal encoding injected at each step, possibly to a subset: GoalNCA [snippet only].
- DeepSDF (Park et al., CVPR 2019) as the standard continuous SDF shape code [snippet only: title/venue seen; not read].
Transfer lesson: you need a 'shape code' budget. Options to compare: (a) latent vector from a shape encoder, broadcast to all, (b) per-agent local lookup in a distance-transform grid, (c) quadtree. (a) is the diffusion-natural one but must be small and static; (b) is what real swarms use and is Stage 2 realistic.

### 2.5 Metrics and datasets

Metrics seen in verified sources:
- Sun et al. 2023: coverage rate (fraction of target cells occupied), entering rate (fraction of robots inside shape), distribution uniformity (spread of nearest-neighbour distances), velocity polarization, convergence time and rate [verified].
- Chamfer distance (squared NN gaps, symmetrised) vs Wasserstein/EMD (needs equal-sized sets): a snippet from an implicit-shape paper evaluated Chamfer on 30k points, Wasserstein on 1k [snippet only]. LLM-swarm benchmark (arXiv 2505.04364, "Benchmarking LLMs' Swarm intelligence") scores shape matching with an EMD-style translation-invariant cost [snippet only].
- DiffeoMorph (Pahng, Guan, Fefferman, Hormoz; arXiv 2512.17129, Dec 2025, v2 May 2026) [verified abstract]: agent-based morphing in 3D with a loss on 3D Zernike polynomials treating shapes as continuous distributions, independent of agent order, agent count and global orientation, rotation-aligned but reflection-sensitive. Number of shapes not in abstract.
Lesson: use a bundle: Chamfer + Wasserstein (equal-size resample) for geometry, IoU on rasterised occupancy, coverage / entering rate / NN-distance uniformity from Sun et al., plus time-to-form and per-step displacement. A Zernike-type or occupancy-density loss is order- and N-invariant.

Datasets (all [from memory] unless stated): MPEG-7 CE-Shape-1 (1,400 silhouettes, 70 classes x 20), Google QuickDraw (50M sketches, 345 classes), emoji sets, Kimia silhouettes, procedural blobs (random Fourier/radial-function contours, unions of primitives, morphological ops). I did not find a swarm paper that uses MPEG-7 or QuickDraw; the swarm papers above use hand-picked shapes (starfish, snowflake, letters, star). My search for point-set diffusion on those datasets found none.
Lesson: your procedural generator should span a controlled difficulty axis (number of holes, number of arms/curvature, thin parts, concavity, scale relative to R). Reserve out-of-distribution families (e.g. MPEG-7 silhouettes, hand-drawn QuickDraw) as unseen test sets.

### 2.6 How many training shapes for generalisation?
- No paper I found reports a training-set-size scaling curve for decentralized assembly or swarm shape formation. [search negative]
- Blocks Assemble: training blueprints of 2 to 16 blocks, zero-shot on unseen ones, about 1e9 steps (snippet; abstract verified for zero-shot claim).
- GoalNCA: "sometimes handles unseen scenarios" (snippet only).
- Equivariant GNN flocking: symmetry gives 70% less training data (verified, but different task).
- Image-diffusion analogue (Kamb and Ganguli) suggests generalisation arises from local patch recombination, so diversity of local patterns matters more than the count of global shapes. [inference]
Lesson: plan a scaling curve yourselves: 10^2, 10^3, 10^4 shapes with fixed held-out set; this is a publishable, cheap ablation.

## 3. Pitfalls: ways a diffusion swarm can look good in sim but be unrealisable

1. Global clock / noise schedule. Standard sampling has all N particles at the same denoising time `t`. Real swarms have no shared clock (Huddle is explicitly asynchronous). Test: per-particle random `t` jitter; time-free (autonomous, time-invariant) velocity field; or a locally estimated "progress" feature instead of `t`.
2. Global attention / all-to-all. MADiff-style and DiT denoisers (Flow-Opt, SwarmDiff) attend to all agents. Enforce a hard neighbour mask of radius R, use the same R as the robot sensor, and count message-passing rounds as physical latency (cf. pivoting cubes: more rounds is better but costs time).
3. Global shape code broadcast every step. One-time broadcast of a small code is plausible; a high-dimensional per-step conditioning, or conditioning on the full target point set via cross-attention, is not. Report code size in bytes.
4. Unlimited speed / teleportation. Diffusion steps can move particles far per step. Cap displacement per step and per unit time to the robot speed; make step count the time axis; reject paths that need more than v_max.
5. Straight-line noise-to-data paths. Flow-matching trajectories are straight in particle space from random noise positions and cross the shape, passing through each other. Physical robots start from a physical initial distribution (e.g. a pile, a blob), not N(0, I) across the arena; train on that.
6. Particle identity and ordering. Models trained with per-index targets or learned positional embeddings of particle index assume labelled agents. Require permutation-equivariant architecture and N-/order-invariant losses (Wasserstein, Chamfer, Zernike as in DiffeoMorph); no agent ids as inputs.
7. Collisions and finite size. Point particles overlap freely. Add a repulsive term, projection step (as Flow-Opt's safety filter) or hard-core in training; evaluate minimum pairwise distance and contact rate.
8. Variable and unknown N. Real swarms lose and gain robots (repair). Training with fixed N bakes in density. Train with N sampled over a range (Sun et al. tested robot/cell ratios; Kilobots 1,024 fixed); test at N never seen; make the target a density rather than a count.
9. Global frame and localisation. Sun et al. needed mocap and a shared frame. A denoiser that reads absolute coordinates, or the target's pose in the world frame, silently uses global localisation.
10. Stochastic sampling vs persistence. Diffusion terminates at t = 0; swarms must stay formed and repair. Add a stationary mode (denoise at fixed low noise) and test damage recovery (removal of 10 to 30% of particles); NPA claims this natively.
11. Synchronous updates. All particles update simultaneously with perfect neighbour state. Test random asynchronous order, stale neighbour info (cf. spatio-temporal GNN for delayed states), sensor noise, and dropped messages.
12. Shape-code leakage. If a global target code is available to every particle, the "swarm" is just N independent particles decoding a code; mode collapse or perfect fit is not evidence of collective computation. Include an ablation with the code given only to a fraction of particles (GoalNCA reportedly tolerated this).
13. Evaluation by Chamfer alone. Chamfer is insensitive to uneven density (clumping); also report uniformity and coverage (Sun et al.), otherwise a clump at the boundary looks fine.
14. Compute that cannot run onboard. Multi-step sampling (tens to hundreds of network evaluations) per robot step; choose a microcontroller-feasible network or distil to a few steps.

## 4. Proposed ladder of experiments (INFERENCE, not from the literature)

R0. Reproduction and teachers (sanity). Mean-shift (Sun et al.) and NPA single-shape on your shape sampler as baselines; teammate's MLP-from-teacher as comparator.
- Success: baselines reach coverage above 90% and a Chamfer value that defines your reference error floor; time-to-form recorded.
- Falsifier: if mean-shift already solves nearly all 10k shapes at your N, the learned approach has to win on something else (repair, no per-robot map, compute).

R1. Free 2D particles, global attention, fixed N, shape-conditioned flow matching (upper bound).
- Success: held-out shapes at Chamfer within, say, 1.5x of the training-shape error; sample uniformity near the target's.
- Falsifier: held-out error much worse than train (more than 3x) at 10k shapes means the conditioning/model cannot generalise even without locality; stop and fix before adding constraints.

R2. Same, but local denoiser (neighbour radius R), still shared `t` and fixed shape code.
- Success: error within about 20% of R1 for R at several particle spacings; shape-error versus R curve with a knee.
- Falsifier: error rises sharply for R below shape feature scale and never recovers with more message rounds, or shows wrong global topology (arm count/holes) per the Kamb-Ganguli patch-mosaic prediction. That is a structural limit of locality, not a tuning issue.

R3. Remove global time: per-particle jittered or autonomous (time-free) dynamics; asynchronous updates, stale neighbours.
- Success: error degrades by less than 25% relative to R2 under 50% async and one-step delay.
- Falsifier: collapse without a shared `t`, which means the model needs a clock and the approach needs a physical sync (e.g. a broadcast beacon).

R4. Variable N and damage: N in [N0/2, 2 N0] unseen at test; remove 20% of particles mid-run.
- Success: coverage recovery above 90% within a bounded number of steps; error independent of N within 20%.
- Falsifier: error scales with N or does not recover; compare with mean-shift and NPA, which are claimed to repair.

R5. Shape-code ablation: code dimension, broadcast to only a fraction f of particles, or replaced by a per-robot distance-transform grid / quadtree.
- Success: performance maintained down to a small f or code size competitive with Sun et al.'s tens-of-kB map.
- Falsifier: needs full-resolution code at all particles, i.e. no communication advantage over the mean-shift map.

R6. Training-set scaling: 10^2, 10^3, 10^4 shapes, fixed held-out set including OOD families (MPEG-7-like silhouettes, QuickDraw).
- Success: held-out error saturates by about 10^3 to 10^4; OOD gap shrinks with set size.
- Falsifier: held-out error still falling linearly in log N at 10^4 means the generator is too narrow or the model memorises; broaden the generator before spending compute on Stage 2.

R7. Stage 2: 2D microbots with real constraints: finite-size bodies and collisions (projection/safety filter), speed cap, motion noise (Kilobot-like), range-limited noisy comms, physical start distribution.
- Success: coverage above 85% with zero or rare (below 1%) pairwise overlaps, time-to-form within 2x of mean-shift; teacher-MLP vs generative comparison on held-out shapes.
- Falsifier: generative approach loses to the plain MLP-from-teacher on held-out shapes at equal local info and speed cap, or needs more than a few NN evaluations per physical step. If it only wins on symmetric/multi-modal targets (CoDiMAD argument), report that scoped result.

R8. Stage 3: 3D in simulation (cf. DiffeoMorph, Yue et al. 3D, Omotuyi and Kumar 3D).
- Success: same metrics with volumetric IoU; gravity/support constraints respected.
- Falsifier: locality knee (R2) moves such that required neighbour counts explode in 3D, or support/overhang constraints cannot be learned without an explicit physics layer.

## 5. Open gaps (things I could not establish)

- No paper found on diffusion/flow-matching for swarm shape formation with local sensing. Possibly out there under other keywords ("pattern formation", "programmable matter"); my searches were limited to roughly 20-25 minutes.
- No training-set-size scaling evidence for decentralized assembly; Blocks Assemble numbers (2 to 16 blocks, 1e9 steps) are from a snippet, not read.
- Learned self-assembly of M-Blocks / programmable matter with deep learning was not found; M-Blocks and many 2018-2022 papers were not searched.
- Active-matter / Brownian-particle viewpoint as a generative model: nothing found, only the theory links (Franceschi et al.; Chen-Georgiou-Pavon). Nothing applied.
- Citations I could not fully confirm: Boroson IROS 2017 (title/authors unknown), Omotuyi and Kumar 2024 (second-hand), Wang and Rubenstein T-RO 2020 (volume/DOI), Kilobots Science 2014 (secondary sources only), GoalNCA arXiv ID (taken from a search URL), DeepSDF (title/venue only), MPEG-7 and QuickDraw statistics (from memory), Growing NCA (from memory), the "Strickland et al." planar construction RL paper (snippet mention only, no ID), "Collisionless Pattern Discovery" venue.
- Whether SwarmDiff's reported scalability and risk results are reliable: a third-party review flagged an equation error.
- Hardware details of your microbots (sensing radius, speed, comms) were not given to me, so the constraints in R7 are generic.
