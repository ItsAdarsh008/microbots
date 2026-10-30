# E. Shared global state: consensus primitives, density/mean-field control, hierarchy, time-free denoisers

Compiled 2026-10-29 (topic B). Tags: [verified] = I fetched the arXiv/journal page and read the abstract or page text (NOT the full paper unless stated); [snippet only] = only a search excerpt or secondary write-up; [from memory] = neither. Numbers not seen on a fetched page are marked unknown. Anything labelled "inference" is my reasoning, not a finding of a cited paper.

## 1. Bottom line

1. Every physically credible way to get a shared statistic (average consensus, push-sum, max-consensus, hop-count relay) needs on the order of graph-diameter rounds, and the diameter is small for us: with ~6-spacing radius and N=256 it is a few hops (inference), so a pooled statistic is cheap exactly where the local denoiser already works, and expensive (diameter grows ~1/radius) exactly where it fails.
2. Density/mean-field swarm control is the closest existing analogue of your idea: bins plus a swarm-distribution estimate fed back to local rules (Bandyopadhyay et al. PSG-IMC; Zheng et al. mean-field feedback; the 2026 mass-estimator mean-shift paper). All of them find the estimator quality, not the controller, to be the weak link (estimation error is the explicit robustness term).
3. The time-free-denoiser literature (Sun et al. ICML 2025; EqM; 2026 follow-ups) says the noise level is recoverable from the input only through high-dimensional residual statistics (error O(1/sqrt(d-k))). A local window sees few dimensions, so a clock-free local denoiser is expected to be poor, and a swarm-pooled statistic is the principled replacement for the clock (inference, strongly motivated).
4. Hierarchy is the standard design everywhere (cascaded diffusion, latent diffusion, GoalNCA, Kilobot seed+gradient, mean-shift stored grid): coarse global code + local refinement. Your 8x8 occupancy grid is already the "coarse code"; the new global state is the dynamic counterpart (current density), which is what closed-loop density control uses.
5. No paper found that gives a learned local flow/diffusion swarm denoiser a pooled global channel, and nothing on repair via a broadcast deficit signal; Markov-chain density guidance gets self-repair "for free" from the target-vs-current density mismatch, which is the cleanest theoretical statement of what global info is sufficient.
6. Biggest risk for the physical story: aggregation under loss/asynchrony biases (plain consensus converges to the wrong value under packet drops; hop-count gradients suffer error cascades at 1000 robots), so the pooled statistic should be trained with noise/staleness and bounded bandwidth.

## 2. Per-paper entries

### 2.1 Distributed aggregation primitives

**Gossip-Based Computation of Aggregate Information (Push-Sum)** [snippet only]
Kempe, Dobra, Gehrke. FOCS 2003, pp. 482-491. https://cs.cornell.edu/johannes/papers/2003/focs2003-gossip.pdf (PDF unreadable to my fetcher; details from search excerpts.)
Push-Sum: each node keeps (sum, weight), sends half to a random node each round; sum/weight converges to the average. Excerpts give O(log n + log 1/eps + log 1/delta) rounds on a complete-graph-style random-gossip model, with mass conservation as the proof device. Secondary sources say push-sum tolerates some interleavings that break push-pull averaging, but I found no explicit message-loss theorem.
Transfer: for a well-mixed contact graph, a statistic converges in O(log N) rounds, but swarm contacts are geometric (local), not random, so the relevant bound is the diameter/mixing time of the geometric graph (next entry). Push-sum's (value, weight) pair is a tiny message (a few floats per statistic).

**Randomized Gossip Algorithms** [verified abstract page]
Boyd, Ghosh, Prabhakar, Shah. IEEE Trans. Inf. Theory 52(6):2508-2530, 2006. https://web.stanford.edu/~boyd/papers/gossip.html
Averaging time is governed by the second-largest eigenvalue of the doubly stochastic gossip matrix, equivalently the mixing time of a random walk on the graph; fastest gossip is an SDP, with a distributed subgradient method proposed; studies geometric random graphs (sensor networks). The abstract gives no explicit scaling law. [From memory, unverified: averaging time on a ring/path scales like n^2 and on a 2D geometric graph much better, around n in clock ticks.]
Transfer: spatial swarms are 2D geometric graphs, so diffusive consensus needs ~ (diameter)^2 steps for plain neighbour averaging versus ~diameter for flooding/max. This matters for "progress" (a mean) versus "max residual".

**Consensus and Cooperation in Networked Multi-Agent Systems** [snippet only]
Olfati-Saber, Fax, Murray. Proc. IEEE 95(1):215-233, 2007. DOI 10.1109/JPROC.2006.887293.
Standard reference for agreement under directed links, link/node failures, delays; extends to gossip, sensor fusion. Convergence speed of linear consensus is governed by the algebraic connectivity (lambda_2) of the graph Laplacian [from memory].
Transfer: the slow mode of any linear averaging is the lowest graph-Laplacian eigenvalue, which for a radius-r geometric graph shrinks ~r^2/L^2 (inference). Halving the radius costs ~4x more rounds.

**Distributed Optimization over Time-Varying Directed Graphs** (subgradient-push) [snippet only]
Nedic, Olshevsky. IEEE TAC 60:601-615, 2015; arXiv 1303.2289.
Push-sum on time-varying, uniformly strongly connected directed graphs; each node needs only its out-degree; no knowledge of N or the graph sequence. Rate for the optimisation is sublinear (~ln t/sqrt(t)); I did not verify the constants.
Transfer: relevant because moving particles make the communication graph time-varying and (with range limits/obstruction) possibly directed; push-sum is the right primitive and needs no knowledge of N (useful for variable N).

**Distributed algorithms for consensus under packet-dropping links (ratio/robust consensus)** [snippet only]
Vaidya, Hadjicostis, Dominguez-Garcia et al. arXiv 1109.6392 (Part II: coefficients of ergodicity); related: Hendrickx/Frasca (ECC 2013).
Plain consensus with dropped packets stops preserving the average (the matrix is no longer stochastic) and lands on a wrong value; ratio (push-sum-like) consensus with auxiliary mass-tracking variables converges to the exact average under bounded delays/drops. A snippet attributes to Frasca and Hendrickx the finding that larger networks are more robust in accuracy to losses; not diameter-specific.
Transfer: if bots drop messages, build the shared state from ratio-type statistics (sum + count), or accept bias and train the denoiser with injected loss. Which one wins is an experiment (see section 4).

**Max-consensus / flooding** [from memory]
Max/min propagation (each node keeps the max of itself and neighbours) converges exactly in D rounds, D = hop diameter, is insensitive to repeated or reordered messages, and is monotone (so trivially loss-tolerant if links recover). It fails for averages. No specific citation verified; classic in the consensus/flooding literature.
Transfer: "worst-bot residual" or "any bot not yet settled" is a max-statistic and costs only D rounds and one number; a "progress = mean residual" needs averaging. Max-type progress is more robust than mean-type.

**Programmable self-assembly in a thousand-robot swarm** [snippet only]
Rubenstein, Cornejo, Nagpal. Science 345(6198):795-799, 2014. DOI 10.1126/science.1254295.
1,024 Kilobots (about 3 cm), identical program, local neighbour communication (about 3 robot diameters in the excerpt), seed-initiated: gradient formation (hop-count distance from seed, each robot = 1 + min neighbour value), edge-following, localisation; shapes included star, wrench, letter K; neighbour-based fault handling. Reported physical problems: imprecise locomotion, noisy distance sensing, message loss. Assembly time on the order of hours for the full swarm [from memory].
Transfer: this is the physical precedent for your "hop count" channel. Hop-count is a 1-number-per-bot relay with latency = hop distance x message period. Bots do not need a global clock or map besides the stored shape (here a 2D image + seed).

**Error Cascades in Collective Behavior: A Case Study of the Gradient Algorithm on 1000 Physical Agents** [snippet only]
Gauci, Ortiz, Rubenstein, Nagpal. AAMAS 2017 (Princeton SSR page; not arXiv). https://ssr.princeton.edu/publications/error-cascades-collective-behavior-case-study-gradient-algorithm-1000-physical-agents
The hop-count gradient is simple in theory but very sensitive to rare errors (e.g., a corrupted message) at 1000-agent scale because a low erroneous value propagates and persists via the min+1 rule; they give a model of the error probability as function of rare-error rate and network size, validated with injected errors on a line.
Transfer: a monotone min-type relay has a one-sided failure mode (stuck under-estimates). If the shared state is hop-count-like, test one-sided corruption specifically. Averaging-type states have symmetric, decaying error instead. (Note: arXiv 1708.03341, which a search returns near this topic, is an unrelated eLetter criticising the 2014 paper; do not cite it as the cascades paper.)

**Kilobot: A Low Cost Scalable Robot System for Collective Behaviors** [snippet only]
Rubenstein, Ahler, Nagpal. ICRA 2012, pp. 3293-3298. DOI 10.1109/ICRA.2012.6224638.
$14 parts per robot; vibration locomotion; range-only neighbour sensing; neighbour comms; the system includes an overhead controller that talks to the whole collective for programming. The excerpts did not describe the overhead broadcast bandwidth; [from memory] it is a one-to-all IR broadcast used for start/stop/program, with no per-robot addressing.
Transfer: a physical one-to-all beacon is a standard, cheap component of Kilobot-scale swarms; it supports a broadcast of a few bytes (target grid, a global clock tick, a deficit flag) but it is outside-in, so it breaks the "fully autonomous" claim; the uplink (swarm to beacon) needs a camera/sensor.

### 2.2 Density, mean-field and Eulerian control

**Mean-shift exploration in shape assembly of robot swarms** [verified page text via PMC]
Sun, Zhou, Ma, Li, Gross, Chen, Zhao. Nat. Commun. 14:3476, 2023. DOI 10.1038/s41467-023-39251-5. https://pmc.ncbi.nlm.nih.gov/articles/PMC10264375/
The operator-supplied black/white image is converted to a grayscale grid, stored on every robot (tens of kilobytes); each robot starts with itself as the shape centre and negotiates a shared position/orientation with neighbours via local wireless; robots sense nearby cells as occupied or free; mean-shift moves a crowded robot toward the highest density of nearby unoccupied cells of the shape. 50 physical robots (27-robot room entry, 36-robot maze, 8-robot cargo); 16 to 1,024 robots in simulation with only mild growth in convergence time; at 300 robots convergence is at least 20x shorter than compared methods; coverage above 93% with cell-to-robot ratios 0.45 to 128.06; 7 informed of 128 suffice for consensus on a time-varying shape; without the shape-exploring term, snowflake coverage drops to 75%. Repair: after removing one starfish arm's robots, the swarm re-grew the arm with no goal reassignment. Physical runs used external motion capture.
Transfer: (a) the density that matters is the LOCAL density of free vs occupied target cells (same as your local read of the 8x8 grid) and no global swarm statistic is needed for convergence there; (b) the shape-exploring term, not any global signal, produced repair. This is the benchmark your global state must beat: if local flow matching plus a stored map can already repair, a pooled statistic must show a gain in repair or clock removal specifically, not in nominal formation.

**Meanshift Shape Formation Control Using Discrete Mass Distribution** [verified abstract]
Cai, Gao, Li, Wang, Sun, Lu. arXiv 2602.00980, 1 Feb 2026. https://arxiv.org/abs/2602.00980
Replaces a continuous density by a discrete mass distribution over target sample points; a decentralised meanshift law moves the swarm distribution toward the target using feedback of mass estimates; each robot estimates the mass at every sample point with a designed estimator shown to converge asymptotically to the true global values; handles swarm size changes. The abstract gives no swarm sizes or numbers and does not say what is exchanged.
Transfer: this is the closest published instance of a shared coarse density state (per-sample-point mass) that is estimated in a distributed manner and fed back to a local rule. Its bandwidth is O(#sample points), the same shape as your histogram on a coarse grid.

**Probabilistic and Distributed Control of a Large-Scale Swarm of Autonomous Agents (PSG-IMC)** [verified abstract]
Bandyopadhyay, Chung, Hadaegh. arXiv 1403.4134 (v1 Mar 2014, v3 Sep 2016; submitted to IEEE Trans. Robotics). https://arxiv.org/abs/1403.4134
Space is divided into bins; each agent hops between bins with probabilities from a time-varying Markov chain updated with an estimate of the current swarm distribution; minimises expected transition cost; handles agents joining or leaving; simulations plus quadrotor experiments. Per search excerpt: the feedback uses a Hellinger-distance between current and target distributions and the earlier optimal-transport version (CCA 2014, pp. 498-505, "Probabilistic swarm guidance using optimal transport") loses accuracy when the feedback has estimation error, and obtains the current swarm layout by consensus among neighbours. The abstract gives no swarm or bin counts.
Transfer: the exact architecture you propose (coarse grid histogram of the swarm, read by each bot, computed by neighbour consensus) is the established practice in Markov-chain swarm guidance, and the known failure mode is estimate error.

**A Markov Chain Approach to Probabilistic Swarm Guidance** [snippet only]
Acikmese, Bayard. ACC 2012, pp. 6300-6307; journal version Asian J. Control (2015); a related Auton. Robots paper (DOI 10.1007/s10514-015-9470-z) adds safety constraints.
Each agent follows its own Markov chain whose stationary distribution is the target; no inter-agent communication in the base version; the collective distribution converges to the target and self-repairs after disturbance; journal version offers Metropolis-Hastings and LMI designs, the LMI one with a guaranteed exponential rate.
Transfer: gives the theoretical answer to "what global info is sufficient": none at all, if the target density is known to each agent and agents tolerate stochastic motion (convergence only in distribution, not per-agent identity). Feedback from the current density buys speed and a hard constraint on transition counts, not feasibility. This explains why your repair does not conceptually need a global signal, only a stochastic or exploratory local rule.

**Transporting Robotic Swarms via Mean-Field Feedback Control** [verified abstract]
Zheng, Han, Lin. arXiv 2006.11462 (v1 Jun 2020, v3 Mar 2021; related DOI 10.1109/TAC.2021.3108672). https://arxiv.org/abs/2006.11462
Top-down: mean-field PDE for the swarm density; velocity field designed centrally from density feedback; each robot only senses the velocity field and computes its own command; closed loop is locally ISS with respect to density estimation errors; agent-based simulations.
Transfer: a design where the global state is a density-derived velocity field and each bot reads it at its own position, which is exactly the "each bot reads a field at its position" pattern you already use for the target map. The ISS result formalises that the pooled estimate may be noisy as long as it enters as a bounded perturbation.

**Mean-field models in swarm robotics: a survey** [snippet only]
Elamvazhuthi, Berman. Bioinspiration & Biomimetics 15(1):015001, 2019/2020. DOI 10.1088/1748-3190/ab49a4.
Continuum (ODE/PDE/difference) models whose complexity does not grow with N, used for scalable controller design.
Transfer: citation for justifying the mean-field/histogram state as N-independent (your 256 vs larger swarms).

**Optimal Transport-Based Decentralized Multi-Agent Distribution Matching** [verified abstract]
Lee. arXiv 2601.00548 (v1 2 Jan 2026, v2 23 Feb 2026; listed as IEEE TAC 2026). https://arxiv.org/abs/2601.00548
Decentralised matching to a target distribution through optimal transport; each agent picks a terminal location with local information; sequential weight updates build feasible local transport plans; memory-based correction for intermittent, range-limited communication; convergence guarantees (cycle-wise improvement of a surrogate transport cost). No numbers in the abstract.
Transfer: gives an OT-with-bounded-communication precedent and a "memory for intermittent comms" trick, which maps to a stale-global-state training augmentation.

**Multiscale distributed MPC for Optimal Transport** [verified abstract]
Pascual, Huang, Martinez. arXiv 2610.19390 (14 Oct 2026; ACC 2027 submission, under review). https://arxiv.org/abs/2610.19390
Two levels: an OT problem with terminal penalty for time-varying densities; low-level MPC tracks nominal velocity; agents solve the regularised static dual in a distributed manner; distributional input-to-state stability in 1-Wasserstein. No quantitative results on the page; what each agent must know is not stated there.
Transfer: the same hierarchy (coarse density flow plus local tracker) with a stability proof that tracking errors only perturb the macroscopic flow by a bounded amount. A preprint under review, so treat as provisional.

**Mean Field Multi-Agent Reinforcement Learning** [snippet only]
Yang, Luo, Li, Zhou, Zhang, Wang. ICML 2018, arXiv 1802.05438.
Replaces pairwise interactions with an interaction between each agent and the average action/state of its neighbours or the population; tested on Gaussian squeeze, Ising model, battle game.
Transfer: precedent for feeding a learned policy a mean-field summary rather than all agents; supports pooled-mean input as an inductive bias. [From memory, same idea: Deep Sets (Zaheer et al. 2017) and graph-network "global attribute" (Battaglia et al. 2018), which your GLOBAL_STATE_PLAN.md already cites; I did not verify these two.]

**Density-Driven Progressive Shape Formation for Swarm Robots in Dynamic Environments** [snippet only]
IEEE RA-L 2025 (IEEE Xplore document 11283018; I could not read the page; authors unknown).
Per snippet: operates on local perception only, no communication or identity; hand-designed (not learned); 50 physical robots; a single reference density parameter selects among shape-surrounding, double-layer-contour, and boundary-conformal formations; a news page reports it won an RA-L 2025 best paper award (unverified).
Transfer: another density-feedback precedent that shows a single scalar density reference can steer formation phase. Cite with caution.

### 2.3 Hierarchical / coarse-to-fine generation and NCA with global channels

**Cascaded Diffusion Models for High Fidelity Image Generation** [verified abstract]
Ho, Saharia, Chan, Fleet, Norouzi, Salimans. JMLR 23(47), 2022; arXiv 2106.15282. https://arxiv.org/abs/2106.15282
Base low-resolution diffusion model plus super-resolution diffusion stages; the key trick is conditioning augmentation (perturbing the low-res conditioning input) to avoid compounding error across stages; FID 1.48 at 64x64, 3.52 at 128x128, 4.88 at 256x256 on class-conditional ImageNet.
Transfer: if the global state is produced by one model stage and consumed by the local denoiser, train with a corrupted/stale/quantised version of it (conditioning augmentation), otherwise errors compound. Direct analogue for training the denoiser on noisy histograms.

**High-Resolution Image Synthesis with Latent Diffusion Models** [snippet only]
Rombach, Blattmann, Lorenz, Esser, Ommer. CVPR 2022; arXiv 2112.10752.
Diffusion in the latent space of a pretrained autoencoder, cross-attention for conditioning.
Transfer: a compressed latent (here: 64-number occupancy grid) is enough to carry global structure while the local network supplies detail; reinforces your 8x8 finding. Weak evidence for the dynamic state (not tested there).

**Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems (GoalNCA)** [verified abstract]
Sudhakaran, Najarro, Risi. arXiv 2205.06806 (25 Apr 2022). https://arxiv.org/abs/2205.06806
A goal encoding is fed through a small MLP into a perturbation added to each cell's hidden state at every step, letting a single NCA change behaviour dynamically; performance is retained when only a fraction of cells receive the goal.
Transfer: direct support for the fact that the goal can be broadcast to a subset of bots (informed fraction), and that a small broadcast vector is enough. Consistent with Sun et al.'s result that 7 of 128 informed robots suffice.

**Neural Cellular Automata Can Respond to Signals** [verified abstract]
Stovold. ALIFE 2023; arXiv 2305.12971 (v1 22 May 2023, v2 29 Jan 2024). https://arxiv.org/abs/2305.12971
NCA trained to react to internal (genome) signals and external environmental signals, each applied to a single pixel for one step; the organism takes different forms or colours accordingly.
Transfer: a localized pulse can steer the whole NCA, so a single broadcast event (e.g., "damage") may be sufficient to trigger repair behaviour if the rule is trained with it.

**Attention-based Neural Cellular Automata (ViTCA)** [snippet only]
Tesfaldet, Nowrouzezahrai, Pal. NeurIPS 2022; arXiv 2211.01233.
Self-attention that is "spatially localized yet globally organized"; ViTCA reportedly beats U-Net, UNetCA, ViT baselines in denoising autoencoding on six datasets at similar parameter budgets.
Transfer: evidence that local-attention plus global-coordination structure beats pure local convolution at equal size, in a CA setting (i.e., adding global structure helps, though not necessarily via a pooled channel).

**Growing Neural Cellular Automata** [snippet only]
Mordvintsev, Randazzo, Niklasson, Levin. Distill, Feb 2020, DOI 10.23915/distill.00023.
Per-cell 16-value state with a shared local rule grows and regenerates an image; three training regimes (growing/persistent/regenerating); damage-augmented training gives much stronger regeneration. A later 3D paper (arXiv 2103.08737) is quoted in a snippet as restoring a caterpillar machine to 99% of blocks with repair training versus 30% without (not verified on the page).
Transfer: repair has to be trained in for local-only NCAs, as you found. It does not use a global channel; this is the baseline "local + damage-augmentation" which your global state must beat in repair.

**Neural Particle Automata: Learning Self-Organizing Particle Dynamics** [verified abstract]
Kim, Pajouheshgar, Susstrunk, Jakob, Park. arXiv 2601.16096 (v1 22 Jan 2026, v3 24 Jun 2026). https://arxiv.org/abs/2601.16096
Particle version of NCA: each particle has a position and internal state updated by one shared learned rule; neighbourhood via differentiable SPH operators; morphogenesis, point-cloud classification, texture synthesis; reports NCA-like robustness and self-regeneration. The page shows neither a global signalling channel nor how goals are specified (details live in the HTML version, which I did not read).
Transfer: the particles' hidden state is the only memory and communication medium; nothing global. Your pooled state would be an addition not tested there. Also, cite as the closest existing "Lagrangian NCA" baseline.

**Communication Heterogeneity and Collective Consensus in Neural Cellular Automata** [verified abstract]
Singh. arXiv 2606.21202 (v1 19 Jun 2026, v2 13 Aug 2026), cond-mat.dis-nn. https://arxiv.org/abs/2606.21202
NCA on the density classification task: cells must agree on a global majority no cell can observe; sub-populations with different message protocols read each other through a tunable translation step; protocol mismatch slows consensus and mildly diverges but does not fragment; rules trained on diverse protocols tolerate mismatch, single-protocol rules do not; holds on ring and 2D grid.
Transfer: evidence that learned local rules can implement consensus themselves (no explicit primitive needed) and that training with diverse/mismatched messages buys robustness. Single-author recent preprint, treat as indicative.

**Neural Cellular Automata Learn General Features in their Hidden Channels** [verified abstract, low relevance]
Guichard, Nichele. arXiv 2609.21870 (18 Sep 2026). Hidden NCA channels encode scale-invariant features; not about global signalling. Listed only because a search surfaced it.

**A New Kind of Network? Review and Reference Implementation of Neural Cellular Automata** [verified abstract]
Spitznagel, Keuper. arXiv 2604.24990 (27 Apr 2026, v2 12 May 2026). Review and the NCAtorch library; useful as a pointer to NCA literature, not read for global-channel content.

### 2.4 Time-free denoisers (progress without a clock)

**Is Noise Conditioning Necessary for Denoising Generative Models?** [verified abstract]
Sun, Jiang, Zhao, He. ICML 2025; arXiv 2502.13129 (v1 18 Feb 2025, v2 17 Nov 2025). https://arxiv.org/abs/2502.13129
Removing noise-level input degrades most methods gracefully, a few improve, one variant fails badly (per the excerpt; which one I did not verify on the page); theory bounds the error from noise-level uncertainty and its accumulation over the sampler; a noise-unconditional model reaches FID 2.23 on CIFAR-10.
Transfer: the clock is probably removable for 256 particles in a global denoiser; whether a local one can infer t is the open question, answered by the next entries.

**Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models** [verified abstract]
Wang, Du. arXiv 2510.02300 (v1 2 Oct 2025, v3 13 Oct 2025). https://arxiv.org/abs/2510.02300
Time-invariant equilibrium gradient of an implicit energy; sampling is gradient descent with tunable step size, adaptive optimizers, flexible compute; FID 1.90 on ImageNet 256x256; can denoise partially corrupted inputs.
Transfer: a physical swarm that descends a static field needs no clock and can stop when the field is flat; a "residual norm" threshold is a natural progress signal and natural stopping condition. This pairs with a pooled "mean residual" statistic.

**What Time Is It? How Data Geometry Makes Time Conditioning Optional for Flow Matching** [verified abstract]
Helbling, Gutierrez Hernandez, Hoover, Chau, Ram. arXiv 2605.08344 (8 May 2026). https://arxiv.org/abs/2605.08344
Decomposes the time-blind loss into coupling variance and a time-blindness gap; if data lies near a k-dimensional subspace of R^d, time is recoverable from noisy-interpolant statistics in the d-k orthogonal directions, with a closed-form estimator whose error is O(1/sqrt(d-k)) under a spiked-covariance model; the gap is asymptotically negligible versus coupling variance; verified on CIFAR-10, CelebA-HQ, FFHQ; the lever is the coupling, not time conditioning.
Transfer: the most useful single paper for this project. It predicts that a denoiser seeing a handful of local particles has tiny effective d-k and cannot read the clock, while a pooled statistic over all N=256 particles (d=512 minus low-dimensional shape) restores it. Also says the coupling (how noise points are matched to targets, e.g. OT/minibatch) matters more than conditioning. Preprint, not peer-reviewed as far as I saw.

**The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning** [verified abstract]
Sahraee-Ardakan, Delbracio, Milanfar. arXiv 2602.18428 (20 Feb 2026). https://arxiv.org/abs/2602.18428
Autonomous (noise-agnostic) models do Riemannian gradient flow on a "Marginal Energy"; a learned conformal metric cancels the 1/t^p singularity normal to the data manifold; the Jensen gap in noise (epsilon) prediction amplifies errors and explains failures of blind noise-prediction models; velocity parameterisations are stable (bounded gain).
Transfer: use a velocity (flow-matching) output, as you do, for any clock-free variant; avoid epsilon-prediction in the time-free setting.

**Blind denoising diffusion models and the blessings of dimensionality** [verified abstract]
Kadkhodaie, Pooladian, Chewi, Simoncelli. arXiv 2602.09639 (v1 10 Feb 2026, v2 9 Jun 2026). https://arxiv.org/abs/2602.09639
Blind DDMs never pass the noise amplitude to the network; correctness argued under low intrinsic dimension relative to ambient; formulates noise-level estimation from a single noisy sample as a Bayesian problem; adaptive noise scheme. Page lists an ICML 2025 FoGen workshop acceptance. No numbers on the page.
Transfer: same message as 2605.08344: blind works because of high ambient dimension; local windows lack it.

**Equilibrium Forcing: Adaptive Video Generation Without Noise Conditioning** [verified abstract]
Lillemark, Rojas, Novack, Wang, Du, Ma, Berg-Kirkpatrick, Yu. arXiv 2608.14706 (v1 11 Aug 2026, v2 14 Oct 2026; NeurIPS 2026 per page comments). https://arxiv.org/abs/2608.14706
Drops noise-level conditioning for autoregressive video denoising; separation of the learned field from sampling enables closed-loop inference that adapts to the sample being generated; qualitative claims only on the page.
Transfer: precedent for closed-loop (feedback-driven) sampling with a time-free field, i.e. use a progress statistic to decide step size or when to stop.

### 2.5 Self-repair with global signals

**Photomorphogenesis for robot self-assembly: adaptivity, collective decision-making, and self-repair** [snippet only]
Bioinspiration & Biomimetics (IOPscience article ab2958; authors and details not read).
Appeared in a search for global-signal repair. Only the title was seen; that it uses a light (environmental) field is my guess from "photo". Not read; listed only as a lead.

**Dynamic Reconfiguration of Robotic Swarms: Coordination and Control for Precise Shape Formation** [verified abstract; low relevance]
Prasertying, Garcia, Sritriratanarak. arXiv 2511.10989 (14 Nov 2025; ICACS 2025). Geometric-model algorithm for transitions between formations on physical robots. A snippet attributes a hybrid central/decentralised "unfilled cells emit a gradient that outside robots follow" idea to a Nov 2025 arXiv paper; I could not confirm that this is the same paper, so do not cite it for that claim.

**Memory-Efficient 2D/3D Shape Assembly of Robot Swarms** [verified abstract]
Yue, Li, Xu, Ze, Long, Cao, Sun. arXiv 2509.26518 (v1 30 Sep 2025, v2 23 Feb 2026). https://arxiv.org/abs/2509.26518
Replaces the image-grid target of mean-shift methods with a tree representation that hierarchically encodes the shape in 2D/3D, one to two orders of magnitude less memory, two to four times faster shape entry; physical experiments with 6 to 7 UAVs.
Transfer: hierarchical (coarse-to-fine) target encoding is itself an active line; your 8x8 occupancy map is the "coarse level", a quadtree is the natural extension if finer shapes need more than 64 numbers.

Search finding: I found no paper where a broadcast shape-level deficit triggers repair in a swarm. Repair results I found (Sun et al., Acikmese-Bayard, Growing NCA, Kilobots regrowing a tree structure per one snippet) all rely on local rules plus a known target.

## 3. Physical cost of each kind of global state (inference, not from a single paper)

Assume communication radius R in particle spacings, domain extent L spacings, hop diameter D roughly L_diag / R (geometric graph). With N=256 on a roughly square layout of side L ~ 16 spacings (my assumption, adjust to the real arena), D is about 4 at R=6, about 8 at R=3, about 11 at R=2.

| Global state | Mechanism | Rounds to settle | Per-bot message | Main failure modes |
|---|---|---|---|---|
| Pooled mean/covariance/centroid (progress, mean residual) | push-sum or neighbour-averaging consensus (value, weight) | flooding-like ~D rounds to spread, but neighbour averaging to eps accuracy ~ D^2 log(1/eps) (diffusive); push-sum on well-mixed graphs ~log N | a few floats per statistic (value and weight) | bias if packets dropped (non-ratio consensus); stale values while swarm moves; slow at small R; needs consistent N or weights |
| Max/min statistic (worst residual, "anyone unsettled") | max-consensus / flooding | exactly D rounds | 1 float | stale max persists after cause disappears (needs expiry/epoch); one corrupted high value contaminates everyone |
| Coarse density histogram (G x G bins) | per-bin averaging or per-bin push-sum; or each bot writes its bin into a gossiped table | ~D (flood) to D^2 (diffusive) per refresh | G^2 numbers (64 for 8x8; 16 for 4x4); can send only changed bins | staleness (swarm moves faster than refresh), bin-boundary aliasing, needs shared coordinate frame (Sun et al. negotiate one; Kilobots use a seed), bandwidth grows with G^2 |
| Hop-count / gradient relay from a seed or from a "deficit" source | min+1 relay | D rounds, gradient latency = hop distance x message period | one integer | needs a seed/source; error cascades at 1000 robots (Gauci et al.); one-sided stuck low values; breaks when graph disconnects |
| Broadcast beacon (one-to-all) | overhead IR/RF/light | 1 round | few bytes to ~100 bytes depending on the radio | single point of failure; needs infrastructure; provides clock and target but not feedback unless there is an uplink (camera) |
| Learned grid fields (local fields written by bots and diffused) | pheromone-like diffusion on a virtual or real medium | diffusive, ~D^2 | field value at own position | needs a medium or virtual-stigmergy layer; identical to the histogram case if virtualised |
| Clock (the thing we want to replace) | beacon tick or synchronised timers | 1 | 1 number | drift, loss of sync, reset after damage |

Notes: (i) Plain linear consensus slows as the Laplacian eigenvalue lambda_2 shrinks (~R^2/L^2), so small R is the regime where local denoisers already fail and global state is slowest to arrive; (ii) bandwidth for 8x8 histogram is the same as the stored target (64 numbers), so it is not the bottleneck, latency is; (iii) the swarm moves during the consensus, so the "state" is always D rounds old; to be physical, the sim must apply a delay of ~D message periods to the pooled state.

## 4. What to expect (inference, for our experiments)

1. A pooled progress statistic (mean residual or noise-level estimate over all bots) should let you remove the clock at small loss; a time-free LOCAL denoiser with no pooling should degrade more, because the identifiable noise clock lives in d-k high-dimensional directions (2605.08344, 2602.09639). Test: local radius {2,3,6} x {clock, no clock, pooled progress}. Prediction: the gap "no clock, no pool" vs "clock" grows as R shrinks, and pooled progress closes most of it.
2. A coarse density histogram (4x4 to 8x8) of the current swarm should help formation at small radii (where Markov-chain/mean-field literature says global density is the right feedback), mainly by resolving "who is over-/under-filling which region" and fixing mass imbalance, with smaller gain at R=6, where the local model already nearly matches all-to-all. Expect the gain to concentrate on mass-transport errors (wrong amount of mass per region), not on fine boundary errors.
3. Repair: per Sun et al. and Growing NCA, local rules plus damage-augmented training repair already. A pooled histogram should add something only if damage creates a large-scale imbalance (e.g., a lost region); expect a modest repair gain and, importantly, repair WITHOUT damage-augmented training for the pooled model is a cleaner test of whether the global state carries the "deficit" information (Acikmese-Bayard: target-vs-current density mismatch is sufficient information).
4. Staleness and noise: train the global input with delay (D steps), bin dropout, additive noise and quantisation (conditioning augmentation, 2106.15282). Prediction: models trained on exact oracle histograms degrade sharply when the histogram is delayed or noisy; augmented ones degrade gracefully. Add a one-sided corruption test (stuck-low values) if using hop-count-style states.
5. Mean-type vs max-type progress: max-type (worst residual) is cheaper (D rounds, 1 float) and more robust; mean-type is smoother. Expect mean-type to give the better denoising and max-type the better stopping/termination signal.
6. Variable N and missing bots: normalised histograms (density, not counts) plus push-sum weights keep the state N-free; expect good generalisation to N != 256 only if training varies N.
7. Coupling: 2605.08344 suggests changing the noise-to-data coupling (e.g., OT/minibatch matching) may matter more than any conditioning; run this as a control against the global state, to avoid attributing gains to the pooling that come from different pairing.
8. Distributional convergence vs per-particle identity: Markov-chain guidance converges in distribution; if your metric is per-particle matching, the density state will not fix identity ambiguity, only mass allocation.

## 5. Open gaps

1. I found no work that adds a pooled/global channel to a learned local flow-matching or diffusion swarm denoiser (consistent with note C), and no learned analogue of Hellinger/Markov-chain feedback. Direct novelty claim is plausible but based on search only.
2. No quantitative round counts for consensus on 2D geometric graphs at N=256 were verified; the Boyd et al. scaling and Push-Sum theorem were not readable from the fetched pages (PDF extraction failed). Section 3 numbers are inference. Needs the Boyd 2006 paper's Table/Theorem for geometric random graphs before quoting scaling exponents.
3. Kilobot paper (Science 2014) details (bandwidth of the message, assembly time) were not fetched; Nagpal-lab details are from snippets and memory.
4. Sun et al. 2023: full text was read through the PMC rendering of the page, but I did not extract the communication range or message size per robot (not stated in what I saw).
5. Density-Driven Progressive Shape Formation (RA-L 2025) and Photomorphogenesis (Bioinspir. Biomim.) were not read; treat as leads only.
6. For the time-free literature, whether it holds for a LOCAL window of a swarm is untested by any of these papers (all are image or video models with large d); the d-k argument is my extrapolation.
7. The physical question of how a swarm obtains an exact histogram while moving (delay/error budget) has no clean reference here; Bandyopadhyay et al. note the estimation-error problem but I did not read their numbers.
8. Not covered: mean-field games proper (Lasry-Lions), Wasserstein-gradient-flow swarm control beyond the above 2026 abstracts, and sector-specific hierarchical NCA with an explicit global channel; searches did not return a paper with a broadcast hidden channel in NCA.
