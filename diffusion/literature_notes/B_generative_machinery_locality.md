# B. Generative machinery and locality: literature notes

Compiled 2026-10-29 for the microbot-swarm diffusion / flow-matching project (Stage 1: 2D point particles to 2D shapes).
Tags: [verified] = arXiv abs page fetched and title/authors/dates checked; [snippet only] = only a search excerpt seen; [from memory] = neither.
Anything under "Transfer lesson" or in sections 3-4 is my inference, not a claim of the cited paper. Abstract pages mostly carry no numbers; where a number is given it is from the abstract.

## 1. Bottom line

1. Use flow matching with a minibatch-OT (or hybrid OT/independent) coupling. OT pairing removes path crossings, which gives straight, non-crossing, near-collision-free trajectories and the best chance of few steps. [inference built on Tong 2023, Pooladian 2023, Hui 2025]
2. Locality is not a handicap for denoisers. Two lines of work (Kamb and Ganguli 2025; Niedoba et al. 2025) show image denoisers behave like local patch denoisers, and Lukoianov et al. (2025) argue this locality comes from the data statistics, not from the architecture. A radius-r particle denoiser is therefore plausible, provided the target map carries the global information.
3. A shared clock is probably not needed: Sun et al. (ICML 2025) and Helbling et al. (2026) show time conditioning is nearly optional, and Equilibrium Matching (2025) makes the dynamics autonomous (gradient descent on a learned landscape). Caveat: the identifiability argument relies on high dimension, and a local denoiser sees a low-dimensional neighbourhood. This is untested for swarms.
4. The right architecture family is a permutation-equivariant, radius-graph message-passing or neighbourhood-attention network with one shared rule (EGNN / EPiC / NCA / NPA lineage), with target-map features sampled at the particle's own position.
5. For few steps, MeanFlow / shortcut / consistency-style training is the main lever, but all are demonstrated only on images with global networks. No paper I found combines them with a local-only, clock-free denoiser, so that is the central gap.

## 2. Per-paper entries

### 2.1 Flow matching, rectified flow, OT couplings

**Flow Matching for Generative Modeling** - Lipman, Chen, Ben-Hamu, Nickel, Le. arXiv:2210.02747 (Oct 2022; ICLR 2023 per my memory, the abs page lists no venue). https://arxiv.org/abs/2210.02747 [verified]
Simulation-free regression of a conditional vector field for continuous normalizing flows; covers diffusion paths as a special case and also OT displacement-interpolation paths. Authors report OT paths train and sample faster than diffusion paths and give better ImageNet likelihood and samples (abstract gives no numbers).
Transfer: the objective is a plain per-sample regression. It can be applied per particle with the particle's local inputs. The OT path is a straight line, the natural motion for a robot.

**Flow Straight and Fast: Rectified Flow** - Liu, Gong, Liu. arXiv:2209.03003 (Sep 2022). https://arxiv.org/abs/2209.03003 [verified]
Trains an ODE along near-straight paths between paired samples. The "rectification" (reflow) step produces couplings with non-increasing transport cost, and straighter flows can be simulated with coarse discretisation, down to a single Euler step.
Transfer: straightness is exactly what a physical swarm wants (few rounds, predictable motion). Reflow is a path to fewer steps, but costs a second training stage.

**Improving and generalizing flow-based generative models with minibatch optimal transport (OT-CFM)** - Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio. arXiv:2302.00482, TMLR (v4 Mar 2024). https://arxiv.org/abs/2302.00482 [verified]
Generalised conditional flow matching with OT couplings inside each minibatch; no Gaussian source required. Authors report simpler flows, more stable training and faster inference; when the true OT plan is available it approximates dynamic OT.
Transfer: source need not be Gaussian. For particles, the source can be a physically sensible start (e.g. a dispersed cloud), and the particle-to-target-point assignment is chosen by OT within each minibatch.

**Multisample Flow Matching: Straightening Flows with Minibatch Couplings** - Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, Chen. arXiv:2304.14772 (Apr 2023). https://arxiv.org/abs/2304.14772 [verified]
Pairs data and noise via non-trivial minibatch couplings that keep the correct marginals; claims lower gradient variance, straighter flows and good samples with fewer function evaluations (on downsampled ImageNet).
Transfer: for a point set the coupling is literally an assignment problem (Hungarian / Sinkhorn on N-by-N), cheap at N in the hundreds to low thousands.

**Equivariant flow matching** - Klein, Kraemer, Noe. arXiv:2306.15030 (Jun 2023). https://arxiv.org/abs/2306.15030 [verified]
OT flow matching for rotation- and permutation-invariant many-particle systems; reports shorter integration paths and better sampling efficiency than earlier equivariant CNFs.
Transfer: gives the recipe for combining OT coupling with symmetry (align under permutation, and rotation if the shape distribution is rotation-invariant).

**Not-So-Optimal Transport Flows for 3D Point Cloud Generation** - Hui, Liu, Zeng, Fu, Vahdat. arXiv:2502.12456 (Feb 2025). https://arxiv.org/abs/2502.12456 [verified]
Finds exact OT flows struggle as point clouds grow, and that very straight trajectories make the flow hard to learn near the start. Uses an offline approximate-OT coupling mixed with an independent coupling; beats prior diffusion and flow methods on ShapeNet generation and completion.
Transfer: do not push for perfect OT. A hybrid coupling is a sensible default and precomputing the pairs offline is cheap for a 10k-shape procedural dataset.

**Generalized Wasserstein Flow Matching: Nested Couplings for Point Cloud Generation** - Piening, Duong, Steidl. arXiv:2605.08424 (v1 May 2026, v3 26 Oct 2026). https://arxiv.org/abs/2605.08424 [verified]
Treats each point cloud as a measure and builds nested outer/inner couplings; compares exact Wasserstein plans (lower energy, straighter) against sliced and lazy linear plans (more scalable). Point-cloud experiments illustrate a geometry-cost trade-off. Search snippet (not verified on the page) says it uses a permutation-invariant coupling and a transformer without positional embeddings, and that exact couplings scale quadratically in batch size.
Transfer: a menu of coupling cost options if exact minibatch OT becomes the bottleneck.

**Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching** - arXiv:2505.23346 (authors not confirmed). [snippet only]
Argues geometry-based couplings (OT) may not match the trajectories the network finds easy to learn, and proposes a model-aligned coupling. Treat as a pointer, not a cited result.

**Optimal Transport Flow Matching by Design** - Malnick et al. (per snippet), arXiv:2606.04092. [snippet only]
Treats the prior as a design choice so that an OT-optimal coupling is available without computing a plan. Interesting for particles (design the source cloud so the identity coupling is OT-like) but unverified.

**Why straight and non-crossing matters for particles (reasoning, not a single paper):** with independent coupling, many conditional paths cross, the learned marginal field averages conflicting directions and bends, which costs steps. An OT (quadratic-cost) coupling cannot contain crossing pairs, and straight-line interpolation of an OT map keeps distinct particles at distinct positions for t in (0,1) (McCann displacement interpolation; classical fact, [from memory]). This gives collision-freedom in the idealised limit only; a learned field, finite steps and finite robot radius mean you still need a repulsion or safety layer.

### 2.2 Generative models on point clouds and sets

**Diffusion Probabilistic Models for 3D Point Cloud Generation** - Luo, Hu. arXiv:2103.01458, CVPR 2021. https://arxiv.org/abs/2103.01458 [verified]
Treats points as particles diffusing from the shape distribution to noise; the reverse chain is conditioned on a shape latent; closed-form variational bound.
Transfer: the "points as particles" framing is the same as ours, but the per-point network reads a global shape latent. That latent is what we must replace with a local, queryable target map.

**3D Shape Generation and Completion through Point-Voxel Diffusion (PVD)** - Zhou, Du, Wu. arXiv:2104.03670 (Apr 2021). https://arxiv.org/abs/2104.03670 [verified]
Combines diffusion with a hybrid point-voxel network; unconditional generation and multi-modal completion.
Transfer: voxel branches give local neighbourhood aggregation on a grid. For swarms, the analogue is a local density / neighbourhood field, which the robots can estimate by counting neighbours.

**Equivariant Diffusion for Molecule Generation in 3D (EDM)** - Hoogeboom, Garcia Satorras, Vignac, Welling. arXiv:2203.17003, ICML 2022. https://arxiv.org/abs/2203.17003 [verified]
E(3)-equivariant denoiser for atom coordinates plus categorical types; better sample quality and training efficiency than previous 3D molecule generators.
Transfer: for 2D shapes, E(2) equivariance (translation, rotation) is a useful inductive bias for a swarm that has no common frame; however, robots with a shared target map may not need rotation equivariance if the map fixes the frame.

**E(n) Equivariant Graph Neural Networks (EGNN)** - Garcia Satorras, Hoogeboom, Welling. arXiv:2102.09844 (Feb 2021). https://arxiv.org/abs/2102.09844 [verified]
Equivariant to rotations, translations, reflections, permutations without higher-order representations; scales to higher dimensions.
Transfer: simplest drop-in message-passing denoiser on a radius graph; output a displacement as a weighted sum of relative position vectors, which is naturally local and translation-equivariant.

**Set Transformer** - Lee, Lee, Kim, Kosiorek, Choi, Teh. arXiv:1810.00825, ICML 2019. https://arxiv.org/abs/1810.00825 [verified]
Attention over sets with inducing points reducing cost from quadratic to linear.
Transfer: inducing points are a global bottleneck, which a swarm does not have; use only as an upper-bound "global" baseline.

**EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion** - Buhmann et al. (10 authors), arXiv:2310.00049 (Sep 2023). https://arxiv.org/abs/2310.00049 [verified]
Deep-sets-based permutation-equivariant layers with linear cost; EPiC-FM is described as the first permutation-equivariant CNF for particle clouds, reported state of the art on top-quark JetNet jets.
Transfer: evidence that flow matching on unordered particle clouds with variable multiplicity works with light-weight equivariant nets; the global pooling inside EPiC layers is the part a swarm cannot do.

**Point Transformer V3** - Wu, Jiang, Wang, Liu, Liu, Qiao, Ouyang, He, Zhao. arXiv:2312.10035, CVPR 2024. https://arxiv.org/abs/2312.10035 [verified]
Replaces KNN with serialised neighbour mapping, widening the receptive field from 16 to 1,024 points, with ~3x speed and ~10x memory gains over PTv2; argues scale matters more than intricate design.
Transfer: relevant to training throughput only. A swarm's neighbour structure is dictated by physics (radius r), not by serialisation.

**Recent point-cloud flow papers seen only in search results:** Hierarchical Flow Matching for 3D Point Cloud Generation (arXiv:2608.05557) and TokenFlow (arXiv:2610.27418). [snippet only] Both use shared per-point networks plus a global code or token bottleneck. Not used in the recipe.

Nothing found specifically on generating 2D point sets with variable N and permutation symmetry; see open gaps.

### 2.3 Locality of denoisers

**An analytic theory of creativity in convolutional diffusion models** - Kamb, Ganguli. arXiv:2412.20292 (Dec 2024, v2 Jun 2025), ICML 2025. https://arxiv.org/abs/2412.20292 [verified]
Locality and translational equivariance alone yield analytic "local score" and "equivariant local score" machines that, after calibrating one time-dependent locality scale, predict outputs of trained ResNets/UNets with median r^2 of 0.95 / 0.94 / 0.94 / 0.96 on CIFAR10, FashionMNIST, MNIST, CelebA. Proposed mechanism: a locally consistent patch mosaic. Pretrained attention UNets are only partially predicted (median r^2 about 0.77 on CIFAR10), so attention adds semantic coherence beyond the local rule.
Transfer: the locality scale is time-dependent (large at high noise, small at low noise). A fixed radius r is therefore a mismatch at the early stage; the target map must supply the coarse structure at high noise. Also: attention-based global coherence is exactly what we give up.

**Towards a Mechanistic Explanation of Diffusion Model Generalization** - Niedoba, Zwartsenberg, Murphy, Wood. arXiv:2411.19339 (Nov 2024), ICML 2025 spotlight. https://arxiv.org/abs/2411.19339 [verified]
Network denoisers generalise via localised denoising; patch-based training-free denoisers combining local empirical denoisers reproduce network behaviour and reach lower MSE than earlier hand-built denoisers (no numbers in the abstract).
Transfer: supports training a denoiser whose input is only a local patch; it also suggests a cheap training-free baseline (local empirical denoiser over dataset shapes) to compare against.

**Locality in Image Diffusion Models Emerges from Data Statistics** - Lukoianov, Yuan, Solomon, Sitzmann. arXiv:2509.09672 (Sep 2025, v2 30 Oct 2025 per the abs page), NeurIPS 2025 poster (per search snippet). https://arxiv.org/abs/2509.09672 [verified]
An optimal linear denoiser already shows locality similar to deep denoisers; locality arises from pixel correlations in the data and tracks principal components on specialised datasets; an analytic denoiser built on this matches deep model scores better than earlier hand-crafted ones.
Transfer: locality of the denoiser reflects the correlation length of the data. Our shapes are procedurally generated, so we can measure the correlation length and choose r accordingly; shapes with long-range structure (thin long bars) will need target-map help.

**Neighborhood Attention Transformer (NAT)** - Hassani, Walton, Li, Li, Shi. arXiv:2204.07143, CVPR 2023. https://arxiv.org/abs/2204.07143 [verified]
Sliding-window attention restricted to nearest neighbours; linear cost; keeps translational equivariance unlike Swin windows; NAT-Tiny 83.2% ImageNet top-1, 51.4 COCO mAP, 48.4 ADE20K mIoU.
Transfer: the pixel-grid version of "attention only within radius r". On a particle set the equivalent is attention over a radius graph, which is also what NPA's SPH-style operators do.

### 2.4 Diffusion meets cellular automata / self-organisation

**Neural Particle Automata: Learning Self-Organizing Particle Dynamics** - Kim, Pajouheshgar, Süsstrunk, Jakob, Park. arXiv:2601.16096 (v1 22 Jan 2026, v3 24 Jun 2026). https://arxiv.org/abs/2601.16096 [verified]
Extends NCA to moving particles with one shared learned rule; neighbour perception uses differentiable SPH operators with CUDA kernels; evaluated on morphogenesis, point-cloud classification and particle texture synthesis; keeps NCA robustness and self-regeneration. The abstract gives no numbers.
Transfer: this is the architecture to reuse as the denoiser body; the new part is the training objective (flow matching instead of per-shape rollout) and the target-map conditioning.

**Frequency-Time Diffusion with Neural Cellular Automata (Diff-NCA, FourierDiff-NCA)** - Kalkhof, Kuehn, Frisch, Mukhopadhyay. arXiv:2401.06291 (Jan 2024, v2 May 2024). https://arxiv.org/abs/2401.06291 [verified]
NCA used as the denoiser inside a DDM. Diff-NCA (331k params) generates 512x512 pathology slices; FourierDiff-NCA (1.1M params) reaches FID 43.86 vs 128.2 for a 3.94M-param UNet. The Fourier stage is added because purely local communication lacks early global structure.
Transfer: strongest direct evidence that a purely local shared rule can act as a diffusion denoiser, and the authors' own fix for the weakness (global info early) corresponds to our global target map. Note the FID comparison is on one dataset and against a small UNet, so it is weak evidence for general quality.

**MedSegDiffNCA** - arXiv:2501.02447 (Jan 2025; authors not fetched). [snippet only] NCA as noise predictor inside a diffusion segmentation model, reported Dice 87.84% with 60-110x fewer parameters than UNets. Segmentation, not generation; low priority.

**An Organism Starts with a Single Pix-Cell: A Neural Cellular Diffusion for High-Resolution Image Synthesis** - arXiv:2407.03018 (authors not confirmed here; snippet says Elbatel et al.). [snippet only] Pointer only.

**Growing Neural Cellular Automata** - Mordvintsev, Randazzo, Niklasson, Levin. Distill, 11 Feb 2020, doi 10.23915/distill.00023, https://distill.pub/2020/growing-ca/ . [snippet only] (No arXiv version.) Differentiable local update rule trained by backprop through time to grow and regenerate a target pattern.
Transfer: the single-target ancestor of our work; the new idea is generalising across 10k+ targets.

**Distributed Swarm Morphogenesis with Neural Cellular Automata** - Verma, Ferrante, Tzes. arXiv:2610.31900 (v1 22 Oct 2026, v2 26 Oct 2026), cs.RO. https://arxiv.org/abs/2610.31900 [verified]
NCA plus site assignment and collision-aware control; robots in the formation share local NCA states to find attachment sites; offline sparse-retention fine-tuning and false-positive loss; scales to "several hundred robots" in simulation, validated on Crazyflies. Not a diffusion paper, but the closest swarm-hardware precedent and likely belongs in the other researchers' topics.

I found no paper that uses a diffusion or flow-matching objective with a shared local rule to assemble many target shapes with particles. [negative search result, not proof of absence]

### 2.5 Noise-unconditional / time-free denoisers

**Is Noise Conditioning Necessary for Denoising Generative Models?** - Sun, Jiang, Zhao, He. arXiv:2502.13129, ICML 2025 (v2 Nov 2025). https://arxiv.org/abs/2502.13129 [verified]
Most denoising generative models degrade gracefully without noise conditioning, a few improve; theory bounds the error from dropping it; a noise-unconditional model reaches FID 2.23 on CIFAR-10.
Transfer: drop the clock input. The network must infer progress from the state, which for a local denoiser means from local neighbour statistics.

**Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models** - Wang, Du. arXiv:2510.02300 (Oct 2025). https://arxiv.org/abs/2510.02300 [verified]
Learns the equilibrium gradient of an implicit energy landscape instead of time-dependent dynamics; sampling is gradient descent with adjustable step size, optimiser and compute; reports FID 1.90 on ImageNet 256x256. No venue listed on the page.
Transfer: the most swarm-compatible dynamics seen: autonomous, stops at equilibrium (the formed shape), and no shared clock. Adaptive compute also lets robots stop when locally settled (a local stopping rule is still to be designed).

**What Time Is It? How Data Geometry Makes Time Conditioning Optional for Flow Matching** - Helbling, Gutierrez Hernandez, Hoover, Chau, Ram. arXiv:2605.08344 (8 May 2026). https://arxiv.org/abs/2605.08344 [verified]
Decomposes the time-blind FM loss into coupling variance and a time-blindness gap; shows time is recoverable from noisy interpolants at rate O(1/sqrt(d-k)) when data lie near a k-dim subspace in ambient dimension d; on CIFAR-10, CelebA-HQ, FFHQ, changing the coupling matters more than removing time.
Transfer: two lessons. (a) The coupling is the lever, which matches the OT recommendation. (b) The recoverability argument needs large d - k. A particle that sees only a handful of neighbours has a small effective d, so the argument does not carry over automatically (my inference).

### 2.6 Few-step generation

**Consistency Models** - Song, Dhariwal, Chen, Sutskever. arXiv:2303.01469, ICML 2023. https://arxiv.org/abs/2303.01469 [verified]
Maps any point on an ODE trajectory to its endpoint; one-step FID 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64; distillation or standalone training.
Transfer: self-consistency is a property of a single trajectory, so it can be enforced per particle, but the time input is built into the formulation.

**One Step Diffusion via Shortcut Models** - Frans, Hafner, Levine, Abbeel. arXiv:2410.12557 (Oct 2024, v3 Jun 2025). https://arxiv.org/abs/2410.12557 [verified]
Single network, single training phase, conditioned on noise level and desired step size; reported to beat consistency models and reflow across step budgets; step budget can be chosen at inference.
Transfer: step-size conditioning is a natural fit for "deploy with K physical rounds" because K can be chosen at deployment. It does require a shared step size d, an agreed global constant rather than a clock.

**Mean Flows for One-step Generative Modeling** - Geng, Deng, Bai, Kolter, He. arXiv:2505.13447 (May 2025; NeurIPS 2025 per search snippet). https://arxiv.org/abs/2505.13447 [verified]
Learns average velocity over an interval via an identity linking it to instantaneous velocity; no pretraining, distillation or curriculum; FID 3.43 at 1-NFE on ImageNet 256x256.
Transfer: one-step is a poor fit physically (a single jump by the average velocity ignores collisions and the finite step a robot can take), but 2-4 step MeanFlow is a plausible way to cut rounds.

**Progressive Distillation for Fast Sampling of Diffusion Models** - Salimans, Ho. arXiv:2202.00512, ICLR 2022. https://arxiv.org/abs/2202.00512 [verified]
Repeated halving of sampler steps; FID 3.0 on CIFAR-10 in 4 steps, starting from samplers with up to 8,192 steps.
Transfer: a robust fallback when one-stage methods are unstable; each halving is a separate stage.

### 2.7 Sparse / radius-limited attention and GNN denoisers

See NAT (2.3), EGNN and EPiC (2.2), NPA and Diff-NCA (2.4). Additional pointers I could not verify: no paper found that trains a radius-limited transformer as a flow-matching denoiser on point sets. [negative search result]

## 3. Recommended recipe for Stage 1 (inference, not from any single paper)

- **Loss:** conditional flow matching (velocity regression) as in Lipman 2023 / Tong 2023. Optionally try the Equilibrium Matching target if time-free dynamics prove hard to train.
- **Coupling:** minibatch OT assignment (Hungarian or Sinkhorn on squared distance) between start cloud and target-sampled points, mixed with a fraction of independent pairs as in Hui 2025. Precompute offline for each of the 10k shapes. Check empirically that predicted trajectories rarely come within a collision radius; add a short-range repulsion term in deployment regardless.
- **Source distribution:** try both Gaussian noise and a physically plausible dispersed cloud (e.g. uniform in a disk); Tong 2023 permits non-Gaussian sources.
- **Architecture:** one shared network applied per particle, permutation-equivariant, with inputs limited to (a) relative positions and optionally velocities of neighbours within r, (b) the target map value and its gradient (and perhaps a few multi-scale samples) at the particle's own position. Start with EGNN-style message passing or an NPA/SPH operator; try radius-graph (neighbourhood) attention as a stronger variant. Output a 2D displacement. Variable N then comes for free.
- **Locality:** hard radius cutoff on the graph, plus a measured data correlation length (Lukoianov 2025) to pick r. Expect the locality scale to need to shrink as the state converges (Kamb 2025), so give the early phase a coarse (large-scale, blurred) target map and the later phase a fine one, or feed multiple blur scales of the map.
- **Time conditioning:** none (Sun 2025, Helbling 2026), with a check that the local network can infer progress from neighbour spacing and from target-map residual. If it cannot, add a local proxy (e.g. local density mismatch) rather than a global t.
- **Steps:** train at a fixed, small step budget (e.g. 8-16 rounds), then evaluate 4-8 with shortcut-style step-size conditioning or a 2-4 step MeanFlow; use progressive distillation only if those are unstable.
- **Baselines:** (i) global-attention denoiser (upper bound on quality); (ii) training-free local empirical denoiser (Niedoba-style); (iii) NPA trained on single shapes.

## 4. Open gaps

1. No work I found trains a flow-matching / diffusion denoiser with a strict radius-r receptive field on unordered point sets. All point-cloud models above use a global code, global attention or global pooling.
2. Locality theory (Kamb, Niedoba, Lukoianov) is for image grids with fixed patch size; no equivalent for point sets with varying N and no grid.
3. Time-free identifiability (Helbling 2026) needs high ambient dimension; whether a handful of local neighbours suffices is open and is the highest-risk assumption of the plan.
4. Consistency / shortcut / MeanFlow were shown with global networks and a shared t or step size; how to impose a step-size constant without a shared clock is unspecified.
5. No guarantee in the literature that a learned flow field (as opposed to the exact OT map) is collision-free at finite steps; the physical safety layer is outside the generative-modelling papers.
6. Nothing found for 2D point-set flow matching with variable N specifically (searched once with "extended" mode; one more targeted search of "set diffusion" or "point process flow matching" would be worthwhile).
7. Citations not independently verified: arXiv:2608.05557, 2610.27418, 2606.04092, 2505.23346, 2501.02447, 2407.03018, and the Distill Growing NCA article. Venue statements marked "per search snippet" (MeanFlow at NeurIPS 2025; Lukoianov at NeurIPS 2025) were not confirmed on the arXiv pages.
