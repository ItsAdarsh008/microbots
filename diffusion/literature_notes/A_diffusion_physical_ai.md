# A. Diffusion and flow matching in physical AI / robotics

Researched 2026-10-29. Tags: [verified] = I fetched the arXiv abs/html page and read title, authors, date and content; [snippet only] = only a search excerpt seen; [from memory] = neither. Where a page did not state a number (steps, hardware) I say "not stated" rather than guess.

## 1. Bottom line

1. Nobody has published "a swarm of identical, locally-sensing agents is the thing being denoised by a decentralized generative model." The closest are centralized or semi-centralized multi-robot diffusion planners (SwarmDiff, Train-Small Deploy-Large, Flow-Opt, Gen-Swarms). I found no diffusion work on modular/self-reconfigurable robots (only soft-robot and manipulator design).
2. Gen-Swarms (flow matching on 2,048-point ShapeNet clouds where each point is a drone, ORCA collision avoidance inside the sampler) is the single closest precedent to our Stage 1. It is simulation only and was not run on drones.
3. Few-step generation is solved enough for control: pi0 uses 10 Euler steps, Diffusion Policy 10 DDIM steps, and MeanFlow/self-distilled flow policies reach 1 step with real-robot validation (OGPO on Franka; MP1 and OFP are partly or only simulated, see entries). A 1-5 step budget is realistic for particles too.
4. For hard constraints, the dominant recipe is projection inside the sampling loop (SMD, CADM/SID, Constrained Diffusers) or a sampler-side reactive filter (ORCA in Gen-Swarms, CBF in SafeDiffuser/CoDiG). Projection beats pure gradient guidance in the evidence the authors present, but at 25 steps it is expensive.
5. The only locality-respecting generative-denoising precedents are Diff-NCA (local NCA as the denoiser, images) and SID (per-robot diffusion plus local observations, simulation only). Local-only denoising of a point set toward a global shape is an open gap and probably our main novelty.

## 2. Entries

### 2a. Foundations: policies and planners

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** [verified]
Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song. RSS 2023; IJRR extended version, arXiv 2303.04137 (v5 Mar 2024). https://arxiv.org/abs/2303.04137
Generates action sequences conditioned on observations (conditional denoising, p(A|O)), receding horizon: predict a chunk, execute 8 steps, replan. Training uses 100 diffusion iterations; inference uses DDIM with 10 steps, about 0.1 s on an RTX 3080. Conditioning is FiLM (CNN variant) or cross-attention (transformer variant); vision features are computed once per decision, not per denoising step. Real robots: UR5 Push-T, sauce pouring and spreading, mug flipping, bimanual egg beater, mat unrolling, shirt folding.
Transfer: condition on a fixed encoded context and re-run the cheap denoiser; the paper found action horizon 8 best, i.e. replan often. For a swarm, "chunk" could be a short displacement plan per particle, re-denoised every few ticks.

**Planning with Diffusion for Flexible Behavior Synthesis (Diffuser)** [verified]
Janner, Du, Tenenbaum, Levine. ICML 2022, arXiv 2205.09991. https://arxiv.org/abs/2205.09991
Denoises whole state-action trajectories jointly; classifier guidance and inpainting are the two ways to impose goals and constraints at test time. Mostly simulated benchmarks. (Step count not read; I recall about 20 but that is [from memory].)
Transfer: inpainting is the simplest hard-ish constraint mechanism: clamp known coordinates at every step. For us: pin already-placed/anchor particles, denoise the rest.

**Is Conditional Generative Modeling all you need for Decision-Making? (Decision Diffuser)** [snippet only]
Ajay, Du, Gupta, Tenenbaum, Jaakkola, Agrawal. ICLR 2023, arXiv 2211.15657. https://arxiv.org/abs/2211.15657
Return/constraint/skill-conditioned diffusion over state sequences with classifier-free guidance, plus an inverse dynamics model to get actions. Lesson: generate the desired configuration (states), then use a separate small low-level module to produce actions. This matches a "diffusion proposes target positions; local controller moves the microbot" split.

**pi0: A Vision-Language-Action Flow Model for General Robot Control** [verified]
Black et al. (Physical Intelligence), arXiv 2410.24164 (Oct 2024, v4 Jan 2026). https://arxiv.org/abs/2410.24164
Flow-matching action expert (about 300M params, 3B PaliGemma backbone), 10 forward-Euler integration steps (dt 0.1), action chunks of H=50, the 10 action passes cost about 27 ms of a 73 ms on-board total on an RTX 4090. Seven real robot configurations (UR5e, bimanual, Franka, Trossen/ARX/AgileX mobile). Conditioned on images, language and proprioception through the VLM prefix.
Transfer: 10 Euler steps is enough for continuous outputs at deployment; the expensive part is the observation encoder, which is computed once, not per step. A shape-conditioning embedding can be computed once.

**Real-Time Execution of Action Chunking Flow Policies (RTC)** [snippet only]
Black, Galliker, Levine. arXiv 2506.07339 (Jun 2025, v2 Dec 2025). https://arxiv.org/abs/2506.07339
Inference-time inpainting: freeze the actions that will execute during inference delay and regenerate the rest, no retraining; 12 Kinetix tasks and 6 real bimanual tasks. Follow-up 2512.05964 (training-time action conditioning) also seen as snippet. Lesson: warm-starting from the previously executed plan is how to keep consecutive generations consistent, which matters for a continuously running swarm.

### 2b. Few-step / one-step generation for real-time control

**One Step Diffusion via Shortcut Models** [verified]
Frans, Hafner, Levine, Abbeel. arXiv 2410.12557 (Oct 2024, v3 Jun 2025). https://arxiv.org/abs/2410.12557
One network, one training phase, conditioned on noise level and desired step size, so you can sample with 1 step or many and choose the budget at inference. Images, not robots. Lesson: pick this (or MeanFlow) over two-stage distillation if we want to trade steps for quality at test time.

**MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation** [verified]
Sheng, Wang, Li, Liu. arXiv 2507.10543 (Jul 2025, AAAI 2026). https://arxiv.org/abs/2507.10543
1-NFE action generation from point clouds via MeanFlow identity, CFG for controllability, a dispersive loss to spread embeddings. Adroit and Meta-World; 6.8 ms inference, about 19x faster than DP3; abstract claims real-world validation (hardware details not given on the abs page). Lesson: the dispersive loss is cheap regularization against mode-collapse in few-shot settings.

**OGPO: One-Step Generative Policy Optimization for Real-Time Robot Control** [verified]
Zou, Wang, Wu, Qian, Wang, Li. arXiv 2601.20701 (Jan 2026, v2 Sep 2026, ACM MM 2026). https://arxiv.org/abs/2601.20701
One-step interval-velocity (MeanFlow-style) policy without distillation, then on-policy RL fine-tuning; 5-20x speedup, over 120 Hz, validated on a Franka Panda. Note: a search snippet called this "DMPO / Dispersive MeanFlow Policy Optimization"; the abs page title is now OGPO (renamed between versions). Lesson: generative pretraining then RL fine-tuning with the generative policy as the actor.

**OMP: One-step Meanflow Policy with Directional Alignment** [snippet only]
Fang et al., arXiv 2512.19347. https://arxiv.org/abs/2512.19347
Claims plain MeanFlow acts as a low-pass filter (spectral bias) hurting precision, and adds directional alignment; Adroit, Meta-World and real tasks. I did not open the page; treat claims as unverified. Relevant because fine local placement (precision) is exactly what degrades in one-step generation.

**One-Step Flow Policy (OFP): Self-Distillation for Fast Visuomotor Policies** [verified]
Li, Sun, Chen. arXiv 2603.12480 (Mar 2026). https://arxiv.org/abs/2603.12480
Teacher-free self-distillation (self-consistency loss, self-guided regularization, warm start from temporal action correlation). One step beats 100-step diffusion/flow baselines on 56 simulated tasks, over 100x speedup, and one-step inside pi0.5 beat the original 10-step on RoboTwin 2.0. No real-robot experiments described. Lesson: the warm-start from the previous timestep's output is directly usable for a swarm that is continuously re-denoised.

**Hybrid Consistency Policy (HCP)** [snippet only]
arXiv 2510.26670. https://arxiv.org/abs/2510.26670 (authors not captured)
Short stochastic prefix (25 SDE steps) then a one-step consistency jump approaches an 80-step DDPM teacher on a real robot per the snippet. Also from the snippet: OFP argues Consistency Policy averages over modes, losing sharpness. Treat both as unverified.

**Imitation Learning Policy based on Multi-Step Consistent Integration Shortcut Model** [snippet only]
arXiv 2510.19356 (Oct 2025), authors seen only as "Fang et al." in snippet. https://arxiv.org/abs/2510.19356
Shortcut model plus multi-step consistency loss; two sims, five real tasks; snippet notes one-step quality remains unsatisfactory.

### 2c. Rearrangement / assembly (set of objects denoised jointly)

**StructDiffusion: Language-Guided Creation of Physically-Valid Structures using Unseen Objects** [verified]
Liu, Du, Hermans, Chernova, Paxton. RSS 2023, arXiv 2211.04604 (v1 Nov 2022 titled "Object-Centric Diffusion for Semantic Rearrangement of Novel Objects"). https://arxiv.org/abs/2211.04604
Object-centric transformer plus diffusion; start from randomly scattered objects and jointly denoise goal poses for all objects, conditioned on a partial point cloud and language, then a learned collision discriminator rejects invalid samples. Sim and real rearrangement; 16% higher success on unseen objects than a multimodal transformer baseline. Step count not stated on the abs page.
Transfer: this is the nearest robotics precedent for "a SET is denoised from random scatter to a structure," with a generate-then-filter step for physical validity. Because the set is exchangeable, attention across objects is the natural architecture.

**DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly** [verified]
Scarpellini, Fiorini, Giuliari, Morerio, Del Bue. CVPR 2024, arXiv 2402.19302. https://arxiv.org/abs/2402.19302
Pieces are nodes of a complete graph; positions and rotations are corrupted with noise and a GNN denoises them; SOTA on most 2D/3D reassembly benchmarks, first learned method handling 2D puzzles with both rotation and translation, about 11x faster than the fastest optimization baseline. Vision benchmark, not a robot paper; step count not stated on abs page. Lesson: graph diffusion over an unordered set of pieces works and is fast; also a cautionary note that the graph is complete, i.e. global communication.

**DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics** [verified]
Kapelyukh, Vosylius, Johns. RA-L, arXiv 2210.02438. https://arxiv.org/abs/2210.02438
Zero-shot: object descriptions to a text prompt, DALL-E generates a goal image, the robot rearranges to match it; real-world results. Generates an image, not poses. Lesson (weak): a generative prior can supply a goal layout without any layout data; probably not applicable to our setup beyond motivation.

### 2d. Morphology / design generation

**DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models** [verified]
Wang, Zheng, Ma, Du, Kim, Spielberg, Tenenbaum, Gan, Rus. NeurIPS 2023, arXiv 2311.17053. https://arxiv.org/abs/2311.17053
Diffusion proposes soft-robot geometries; differentiable-simulation gradients guide sampling and co-optimize design and control; fabricated robots shown. Lesson: sim feedback as guidance during sampling steers a pretrained shape prior toward functionally valid shapes. Relevant if we later want the generated shape to be "buildable/repairable."

**Dynamics-Guided Diffusion Model for Sensor-less Robot Manipulator Design** [snippet only]
Xu, Ha, Song. CoRL 2024, arXiv 2402.15038. https://arxiv.org/abs/2402.15038
Per snippet: task-agnostic dynamics network provides gradients to guide design sampling; a design in about 0.8 s. Not modular.

**Modular / self-reconfigurable robots:** my searches found no paper applying diffusion or flow matching to modular robot configuration or self-assembly. The only "diffusion" hits were reaction-diffusion pattern control (Springer 2025, not learned generative). See gaps.

### 2e. Multi-robot trajectory generation (set of agents as the denoised object)

**Gen-Swarms: Adapting Deep Generative Models to Swarms of Drones** [verified]
Plou, Pueyo, Martinez-Cantin, Schwager, Murillo, Montijano. ECCV Workshop (MAAS) 2024, arXiv 2408.15899. https://arxiv.org/abs/2408.15899
Flow matching adapted to 3D point cloud generation where each point is a drone: 2,048 points, conditioned on a 256-d shape latent (PointNet encoder, learned prior) trained on ShapeNet (quantitative only for Airplane). Sampling uses 100 steps in the main comparison; ablation over 5/10/25/50/100 shows shape quality plateaus near 25 steps while more steps improve safety and smoothness. At each step the predicted velocities pass through ORCA, which minimally adjusts them to keep separation (2 m at real scale); authors note this acts like noise on the flow but final shapes stay accurate. ORCA on top of the flow output alone fails to reach the targets. Simulation only; no real drone flights; random initial cloud still contains collisions; positions only.
Transfer: this is our Stage 1 template. Key empirical facts: (a) a reactive collision filter inside the sampler is compatible with flow matching; (b) shape quality saturates at about 25 steps while physical feasibility keeps improving with more; (c) generated shape from noise is a global, centralized sampler.

**SwarmDiff: Swarm Robotic Trajectory Planning in Cluttered Environments via Diffusion Transformer** [verified]
Ding, Jiao, Hu, Zhou, Wu, Mu, Liu. arXiv 2505.15679 (May 2025). https://arxiv.org/abs/2505.15679
Hierarchical: a diffusion transformer generates macroscopic Gaussian-mixture density trajectories (not per-robot paths), conditioned on start/target distributions plus an obstacle ESDF map, with sampling guided by CVaR collision, Wasserstein transport and GP smoothness cost gradients. Per-robot targets then come from minimum-squared-distance assignment and distributed MPC with ORCA. 20/50/100/500 robots in simulation, 10 Wheeltec Ackermann robots with motion capture in the real world. Macro stage about 1 s regardless of robot count; the per-robot stage dominates growth (11.4 s at 500). Diffusion step count not stated. A third-party review flagged a possible missing square root in the CVaR term, so treat the safety claims cautiously.
Transfer: denoising a density/distribution rather than individual identities sidesteps agent labels; the assignment is then a separate step. This is directly relevant to anonymous identical particles.

**Multi-Robot Motion Planning with Diffusion Models (MMD)** [verified]
Shaoul, Mishani, Vats, Li, Likhachev. ICLR 2025, arXiv 2410.03072. https://arxiv.org/abs/2410.03072
Trains single-robot diffusion models only and composes them with multi-agent search (conflict-based) at inference to get collision-free multi-robot trajectories; dozens of robots, simulated logistics scenarios. Lesson: you can avoid learning the joint distribution entirely by training the per-agent model and coupling at inference; supports a per-particle denoiser with local coupling.

**Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models (SMD)** [verified]
Liang, Christopher, Koenig, Fioretto. ICML 2025, arXiv 2502.03607. https://arxiv.org/abs/2502.03607
Projects onto the feasible set (augmented Lagrangian with dual ascent, penalty scaled by zeta; 1.05 suggested) after each of 25 diffusion steps (each with inner SGLD iterations). 3/6/9 robots; runtime up to 586 s at 9 robots on the shelf maps; success 93-100%. Paper argues gradient guidance fails in narrow corridors and rejection sampling is wasteful, but it does not run a direct head-to-head ablation of the three. Simulation only.
Transfer: per-step projection works but is slow and scales badly with robot count; fine for few agents, not for thousands, unless the constraint is local (pairwise distance) and cheap.

**Discrete-Guided Diffusion for Scalable and Safe Multi-Robot Motion Planning (DGD)** [verified]
Liang, Koenig, Fioretto. arXiv 2508.20095 (Aug 2025). https://arxiv.org/abs/2508.20095
Uses discrete MAPF solutions to split the problem into convex subproblems that guide a constrained diffusion model, then a lightweight repair; claims scaling to 100 robots. Lesson: a cheap coarse discrete solution as guidance, then a small repair, replaces full per-step projection.

**Simulation-Informed Diffusion for Decentralized Multi-robot Motion Planning (SID)** [verified]
Liang, Koenig, Fioretto. arXiv 2605.27697 (May 2026). https://arxiv.org/abs/2605.27697
Each robot runs a constraint-aware diffusion model (CADM; 25 sampling steps, projection after each update via augmented Lagrangian) to forecast neighbours' paths from local observations, then plans its own path; horizon up to 64, executes 8 then replans; communication only when no feasible plan is found (0 to about 3 events per robot). Default observation range is half the map; best trade-off at 0.3-0.5 map size, and range 0.1 was tested in the sensitivity study. 3-18 robots quantitatively, a 108-robot / 160-obstacle case shown qualitatively. Simulation only, one A6000.
Transfer: the first decentralized, local-observation diffusion planner I found; the neighbour-forecast-then-act structure is a template for a particle with a small sensing radius. Observation radius matters (larger range gave better success and less communication).

**Train-Small Deploy-Large: Leveraging Diffusion-Based Multi-Robot Planning** [verified]
Singh, Guha, Chang, Acton. arXiv 2604.06598 (Apr 2026). https://arxiv.org/abs/2604.06598
One shared denoiser for all agents (1D conv U-Net with axial attention across the agent axis then time axis, FiLM from scene image, start/goal and agent-count embedding); generates position trajectories for all active agents; curriculum on agent count; trained on 2-4 agents and tested up to 8 in VMAS simulation; 0.04 s sampling; step count not stated. Attention spans all active agents (not neighbour-limited). Weaknesses: success drops as density rises; depends on pre-trained MAPPO data; no real robots. Lesson: a weight-shared, permutation-friendly denoiser with agent-axis attention generalizes mildly beyond the training count, but 4 to 8 agents is far from thousands.

**Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization** [verified]
Idoko, Jadhav, Singh. arXiv 2510.09204 (Oct 2025, v4 Jun 2026). https://arxiv.org/abs/2510.09204
Flow-matching DiT with permutation-invariant robot-position and map encoders samples candidate trajectories; a learned Safety-Filter (differentiable solver, with a network predicting initializations trained self-supervised through it) enforces constraints; tens of robots in tens of ms. Steps and hardware not stated on abs page. Lesson: train the initializer through the differentiable projection so the filter converges in few iterations.

**Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors** [verified]
Idoko, Teja, Krishna, Singh. arXiv 2501.19042 (Jan 2025, submitted to RA-L). https://arxiv.org/abs/2501.19042
CVAE or VQ-VAE (not diffusion) generates swarm trajectories, projected by a safety filter; diverse multimodal feasible trajectories in tens of ms; compares to diffusion guidance and notes cost-term tuning is fragile. Lesson: a non-iterative generator plus a differentiable feasibility projection is a cheap baseline we should compare against.

**MADiff: Offline Multi-agent Learning with Diffusion Models** [verified]
Zhu, Liu, Mao, Kang, Xu, Yu, Ermon, Zhang. NeurIPS 2024, arXiv 2305.17330. https://arxiv.org/abs/2305.17330
Attention-based diffusion over all agents' trajectories; centralized training, decentralized execution, with teammate modeling during execution. Offline MARL benchmarks, no hardware. Lesson: per-agent independent models hurt coordination, joint concatenation hurts sample efficiency; attention across agents is the compromise. Train-Small Deploy-Large explicitly contrasts itself with MADiff's per-agent networks.

### 2f. Hard constraints and safety

**SafeDiffuser: Safe Planning with Diffusion Probabilistic Models** [verified]
Xiao, Wang, Gan, Rus. arXiv 2306.00148 (May 2023; ICLR 2025 per search snippet). https://arxiv.org/abs/2306.00148
Embeds control-barrier-function based "finite-time diffusion invariance" into the denoising process; maze generation, legged locomotion, 3D manipulation; receding horizon. Lesson: constraints can be enforced on the denoising dynamics themselves with a guarantee, not just as a post filter.

**Constrained Diffusers for Safe Planning and Control** [verified]
Zhang, Zhao, Papachristodoulou, Umenberger. arXiv 2506.12544 (Jun 2025). https://arxiv.org/abs/2506.12544
Constrained Langevin sampling (projected, primal-dual, or augmented Lagrangian) added to a pretrained diffusion model without retraining, plus discrete CBFs for online safety; Maze2D, locomotion, PyBullet. Lesson: constraint handling can be added at sampling time to a model trained unconstrained, so we can train once on the 10,000 shapes and switch constraint sets (2D points, then microbot limits) later.

**Constraint-Aware Diffusion Guidance for Robotics (CoDiG)** [verified]
Ma, Bodmer, Carron, Zeilinger, Muehlebach. arXiv 2505.13131 (May 2025). https://arxiv.org/abs/2505.13131
Barrier functions in the reverse process for real-time obstacle avoidance; real miniature racing experiments (platform and step count not on abs page). Lesson: barrier-style guidance can run in real time on small vehicles.

**Joint Model-based Model-free Diffusion (JM2D)** [snippet only]
arXiv 2509.08775. https://arxiv.org/abs/2509.08775 (authors not captured)
Importance sampling to align diffusion planners with non-differentiable safety modules. Unverified.

**Diffusion-based 4D Trajectory Prediction and Distributed Control for UAV Swarms** [verified]
Li, Lu, Li, Zheng. arXiv 2606.31197 (Jun 2026). https://arxiv.org/abs/2606.31197
Diffusion refines residual dynamics inside a distributed nonlinear MPC for a 3-UAV swarm; 34 FPS. Diffusion is a small corrector, not the shape generator; included to show the other way diffusion enters swarm control.

### 2g. Adjacent items relevant to locality (not robotics diffusion)

**Distributed Swarm Morphogenesis with Neural Cellular Automata** [verified]
Verma, Ferrante, Tzes. arXiv 2610.31900 (Oct 2026). https://arxiv.org/abs/2610.31900
NCA plus site assignment and collision-aware control builds lattice shapes with up to several hundred robots in simulation, with an offline fine-tuning step that simulates partial attachment; Crazyflie tests use robot-relative observations reconstructed from external tracking. Not generative diffusion, but it is the closest decentralized-shape-growth baseline to the thing we are comparing against.

**Diff-NCA / FourierDiff-NCA (Frequency-Time Diffusion with Neural Cellular Automata)** [snippet only]
arXiv 2401.06291, https://arxiv.org/abs/2401.06291 (authors not captured)
Per the snippet: diffusion denoiser built as an NCA with neighbour-only communication, about 336k parameters for 512x512 pathology images; plain NCA struggles beyond 64x64 because information spreads one neighbourhood per step, and the Fourier variant adds early global communication. A Nature (npj) version exists ("Parameter-efficient diffusion with neural cellular automata", 2025, snippet only). Direct evidence for the locality bottleneck we will face.

## 3. What this implies for a swarm

Items below marked (inference) are my reasoning, not findings from the papers.

- Step budget (supported + inference). Evidence: pi0 10 steps, Diffusion Policy 10 DDIM steps, Gen-Swarms shape quality plateaus around 25 steps, one-step MeanFlow policies work on real arms. Inference: for 2D point shapes, train with flow matching and use 5-20 steps; add a shortcut/MeanFlow objective only after a baseline works. Gen-Swarms found that fewer steps cost safety/smoothness, not shape accuracy, so physical feasibility, not shape, is what extra steps buy.
- Where the global-vs-local tension sits (inference). Every verified multi-agent denoiser except SID uses all-to-all attention or a centralized sampler (Gen-Swarms, SwarmDiff, Flow-Opt, Train-Small, DiffAssemble's complete graph). Our requirement (radius-limited neighbours, no central brain) means information about a global target shape must propagate by local message passing across denoising steps; the Diff-NCA note that plain NCA denoisers fail at larger scale is a warning that the number of denoising steps times the radius sets how far information travels. Expect to need either a conditioning signal broadcast to every particle (shape code z, as in Gen-Swarms, a minimal "global" input), or noise-level-dependent receptive field.
- Shape conditioning (inference). Gen-Swarms conditions on a 256-d latent; StructDiffusion on language plus point cloud; SwarmDiff on distributions. With 10,000 procedural shapes, a compact per-shape code (or shape-descriptor such as occupancy coarse grid) broadcast to all particles is the lowest-risk choice. Each particle does not need to know its own identity: SwarmDiff's density-then-assign split and Gen-Swarms' unlabeled point clouds both avoid agent IDs.
- Constraints (inference grounded in SMD, SID, Gen-Swarms, Constrained Diffusers). For thousands of particles, full-trajectory projection is too slow (SMD: minutes at 9 robots). The cheap, local, per-step options are pairwise min-distance projection (cf. ORCA inside the sampler in Gen-Swarms) and speed clipping; training-free so they can be swapped when moving to real microbot constraints. Constrained Diffusers shows constraints can be added to an already trained model, supporting a "train unconstrained on shapes, add constraints at sampling" plan. Caveat: Gen-Swarms shows the filter alone, without the flow, fails, and that the filter acts as noise on the flow; test whether projections in the last few steps suffice.
- Replan/warm start (inference). RTC and OFP both exploit the previous output as initialization. A swarm that continuously re-denoises as the shape target changes or particles are lost could start from the current positions at a partial noise level (SDEdit-style) instead of pure noise; this also gives a natural repair mechanism. This is untested in any paper I found.
- Generalization in N (supported weakly). Train-Small Deploy-Large reports mild generalization from 2-4 to 8 agents with a shared denoiser. Nothing reported at hundreds. Procedurally generating shapes at varied particle counts and testing at unseen counts should be part of our eval, since no one has shown it.
- Baselines to include (inference). (a) Swarm-Gen-style one-shot generator plus projection, (b) Gen-Swarms global flow plus ORCA, (c) our NPA models from Kim et al. 2026, (d) the teammate's MLP policy with teacher. The NCA swarm morphogenesis paper (2610.31900) is the nearest decentralized comparator.
- Real hardware (supported). Of the verified multi-agent diffusion/flow items, only SwarmDiff ran physical robots (10 ground robots with mocap). None of the verified swarm/set generative papers addressed microbot-scale physics, so Stage 2 constraints are uncharted.

## 4. Open gaps

1. No verified paper where a decentralized, radius-limited swarm jointly performs the reverse diffusion/flow (each agent holds one point, communicates only with neighbours, noise level as shared clock). SID is the nearest, but is per-robot planning, not per-particle denoising of a global shape.
2. No diffusion/flow work found on modular or self-reconfigurable robots (module graph, attachment, lattice assembly). Searches for this returned only soft robot (DiffuseBot) and manipulator design.
3. Diffusion over unlabelled/anonymous swarms at the thousand-agent scale: largest numbers seen are Gen-Swarms 2,048 points (sim), SwarmDiff 500 (sim), SID 108 (qualitative). Whether quality holds with a purely local receptive field is unknown.
4. Step-count/latency measurements on tiny microcontrollers: all verified results use workstation GPUs. Deployment of a denoiser on a microbot, or running it centrally and broadcasting, is untouched.
5. Hard constraints at scale: projection papers test at most 9-18 robots quantitatively. Local pairwise projection for thousands is, as far as I found, only demonstrated via ORCA in Gen-Swarms (drones, simulation).
6. Not covered or only skimmed due to the time budget: a systematic search of 2026 flow-matching VLA follow-ups (pi0.5, others), diffusion for floorplan/puzzle set generation (PuzzleFusion etc.), and set/point-cloud diffusion from vision (PointFlow, LION) which would likely be the better source for architectures. Citation details for items tagged [snippet only] (authors for 2510.26670, 2509.08775, 2401.06291) were not confirmed.
