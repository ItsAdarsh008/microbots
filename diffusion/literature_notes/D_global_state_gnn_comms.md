# D. Global state in graph/set networks, and communication in multi-agent learning

Researched 2026-10-29. Tags: [verified] = arXiv abs page fetched and title/authors/year confirmed (and, where noted, numbers read from an ar5iv/HTML fetch); [snippet only] = seen in search results, not opened; [from memory] = not checked this session, treat as unverified. Numbers marked "(ar5iv extraction)" came from an LLM-summarised fetch of the HTML paper; re-check the table before putting them in a report.

## 1. Bottom line

1. A pooled global vector is a well-established, cheap fix for the "local graph net cannot see far" problem (virtual node, GN global attribute `u`, DeepSets/EPiC layers, inducing points). It is the standard remedy for over-squashing, but the evidence for it is mostly "task score goes up", rarely "the global channel is actually read".
2. A pooled vector can give centroid, count/density (sum pooling), spread, and a progress/clock scalar. It cannot give "where is the hole" or "which region is under-filled" unless the vector is large enough to encode a map; for that you need a spatial field (grid) or per-bot queries into a latent set (Perceiver-IO style). Continuous-pooling lower bound: latent dim >= set size for exact universal set functions (Wagstaff 2019), so a few floats is a lossy summary by construction.
3. Multi-agent comms literature: a mean-pooled message (CommNet) gave big gains over no communication on tasks that truly need coordination (traffic junction failure 20.6% -> 2.2% MLP), but gains shrink on hard variants and a TarMAC ablation found a 1-scalar message nearly as good as 64-d. So a very small bandwidth can be enough; extra rounds helped more than width.
4. Swarm-control GNN work (Tolstaya et al.) shows a global/centralised controller is the upper bound and local controllers degrade as radius shrinks; multi-hop (K=3-4) recovers most of the gap at small radius. This is the same trade-off as our r ~ 6 spacings finding, and multi-hop depth is the cheap competitor to a global channel.
5. The main risk is that the global channel is ignored or only used as a time/bias input (positive signalling without positive listening, Lowe et al. 2019). Every claim of "global state helps" must be backed by a message-shuffle/zeroing intervention, a capacity-matched local baseline, and a held-out-shape test.
6. Most physically realisable and best precedent for our swarm: a consensus-estimated vector of M masses/densities over fixed sample cells (Meanshift 2602.00980) is exactly a "coarse histogram global state with limited bandwidth".

## 2. Entries

### 2.1 Global attributes and virtual nodes in graph networks

**Relational inductive biases, deep learning, and graph networks** — Battaglia, Hamrick, Bapst et al. (27 authors), arXiv 2018 (1806.01261). https://arxiv.org/abs/1806.01261 [verified]
Defines the Graph Network block with node, edge and global attributes `u`; the global attribute is updated from aggregated node/edge features and fed back to all nodes. Argues for relational inductive bias and combinatorial generalisation. No ablation numbers needed here; it is the formal template for "pooled vector that all nodes write to and read from".
Transfer: our shared state is exactly a GN `u` with a bandwidth cap. Use GN ordering (edges -> nodes -> u -> next block) and decide whether `u` is read before or after node update.

**Neural Message Passing for Quantum Chemistry** — Gilmer, Schoenholz, Riley, Vinyals, Dahl, 2017 (1704.01212). https://arxiv.org/abs/1704.01212 [verified title/authors; the abs page does not mention virtual nodes]
My recollection is that the paper uses a "virtual/master node" variant that adds a latent node connected to all others, but I did not verify this in the text. [from memory for the virtual-node claim]
Transfer: virtual node = our global vector with learned read/write; precedent is old, so no novelty claim.

**On the Bottleneck of Graph Neural Networks and its Practical Implications** — Alon, Yahav, ICLR 2021 (2006.05205). https://arxiv.org/abs/2006.05205 [verified]
Names over-squashing: information from exponentially many distant nodes is squeezed into fixed-size vectors; relieving the bottleneck (e.g. a fully-adjacent last layer) improves long-range tasks without tuning.
Transfer: a radius-r local denoiser with L layers has receptive field ~ L*r; if shape information is far away, it must pass through a bottleneck. Our "r ~ 6 matches global" result is a receptive-field effect, so test depth x radius, not only radius.

**Understanding over-squashing and bottlenecks on graphs via curvature** — Topping, Di Giovanni, Chamberlain, Dong, Bronstein, ICLR 2022 (2111.14522). https://arxiv.org/abs/2111.14522 [verified]
Links over-squashing to negatively curved edges (bottlenecks) and proposes curvature-based rewiring.
Transfer: radius graphs of a uniform 256-particle swarm have no sharp bottlenecks, but thin shapes (a 1-particle-wide line, fragments separated by holes) create exactly the disconnected or bottleneck topology where local information cannot cross. This predicts failure on thin/fragmented shapes at small r.

**Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS)** — Rampasek, Galkin, Dwivedi, Luu, Wolf, Beaini, NeurIPS 2022 (2205.12454). https://arxiv.org/abs/2205.12454 [verified]
Layer = local MPNN in parallel with global attention (optionally linear-cost, e.g. Performer) plus structural encodings; competitive on 16 benchmarks.
Transfer: this is the template for "local branch + global branch" and is what a radius-r GNN plus a few global tokens would resemble. Keep positional/structural encoding (here: absolute map reading) separate from the global channel in ablations.

**On the Connection Between MPNN and Graph Transformer** — Cai, Hy, Yu, Wang, ICML 2023 (2301.11956). https://arxiv.org/abs/2301.11956 [verified]
Shows an MPNN with one virtual node can approximate (a form of) self-attention of a graph transformer, in the opposite direction to the known "transformers emulate MPNNs".
Transfer: one virtual node with enough width can emulate global attention approximately. That argues that a small number of global tokens is plausible, but "enough width" is the whole question (see Wagstaff below).

**Understanding Virtual Nodes: Oversquashing and Node Heterogeneity** — Southern, Di Giovanni, Bronstein, Lutzeyer, 2024 (2405.13526). https://arxiv.org/abs/2405.13526 [verified]
Theory: a virtual node's help with over-squashing depends on graph topology; a standard VN weighs all nodes equally (unlike transformer attention). Proposes a VN variant with node-specific sensitivity at same cost.
Transfer: a uniform mean/sum pool gives each bot equal weight, while our task needs position-dependent reads. Make the read-back query-dependent (attention on the global tokens by position and map value), not a broadcast of the same vector to everyone.

**Local-to-global Perspectives on Graph Neural Networks** — Chen Cai, UCSD PhD thesis, 2023 (2306.06547). https://arxiv.org/abs/2306.06547 [verified abstract only; the VN-on-Peptides-func claim is search snippet only]
Search snippets say adding a virtual node to MPNNs with fixed hyperparameters gives a large gain on LRGB Peptides-func and the best VN model (GatedGCN+RWSE) beats all transformers there. I could not confirm numbers; one leaderboard mirror lists GatedGCN+RWSE+VN at 0.6685 AP vs plain GCN near 0.593 AP. [snippet only]
Transfer: on a long-range benchmark a single VN is a strong, cheap baseline. Use it as the first experiment.

**Local Virtual Nodes for Alleviating Over-Squashing in GNNs** (2508.20597), **RANGE** (2502.13797), **Virtual Nodes Improve Long-term Traffic Prediction** (2501.10048) [snippet only; not opened, authors not recorded]. Pointers only: placing several (local) virtual nodes at bottleneck/high-centrality nodes, or several VNs with positional encodings, beat a single global VN in their reports. Transfer: suggests several spatially-localised summary tokens (a coarse grid) beat one global one.

**Expressivity of aggregators: How Powerful are Graph Neural Networks?** — Xu, Hu, Leskovec, Jegelka, ICLR 2019 (1810.00826). https://arxiv.org/abs/1810.00826 [verified title/authors; abs page does not discuss mean vs sum vs max, my recollection that sum > mean > max for distinguishing multisets is from the paper body] [from memory for that claim]
Transfer: mean-pooling loses count; sum-pooling keeps count but scales with N; concatenate both (as EPiC does) if count/density matters.

### 2.2 Sets, inducing points, latent bottlenecks, registers

**Deep Sets** — Zaheer, Kottur, Ravanbakhsh, Poczos, Salakhutdinov, Smola, NeurIPS 2017 (1703.06114). https://arxiv.org/abs/1703.06114 [verified]
Permutation-invariant functions have form rho(sum phi(x_i)). Transfer: exactly our "pooled vector" channel. Sum of a learned feature map can represent centroid, count, moments.

**On the Limitations of Representing Functions on Sets** — Wagstaff, Fuchs, Engelcke, Posner, Osborne, ICML 2019 (1901.09006). https://arxiv.org/abs/1901.09006 [verified]
Proves that with continuous phi, rho, the latent dimension must be at least the maximum set size for universal representation of set functions. Transfer: a few-float global vector cannot be a faithful summary of arbitrary 256-particle configurations; it is a lossy statistic, so what it carries must be learned/selected (centroid, density moments, progress). Expect it to help only for tasks whose needed information is low-dimensional.

**Set Transformer** — Lee, Lee, Kim, Kosiorek, Choi, Teh, ICML 2019 (1810.00825). https://arxiv.org/abs/1810.00825 [verified]
ISAB: m learned inducing points attend to the set, then the set attends back to them; cost O(nm) instead of O(n^2); PMA pools via learned seed vectors. Transfer: the canonical "few latent tokens read and write" module; m is our bandwidth knob. Keep m small (2-8) and sweep it.

**Perceiver** — Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira, ICML 2021 (2103.03206). https://arxiv.org/abs/2103.03206 [verified]; **Perceiver IO** — Jaegle, Borgeaud, Alayrac et al., ICLR 2022 (2107.14795). https://arxiv.org/abs/2107.14795 [verified]
Latent array cross-attends to inputs (bottleneck), processed by latent transformer; Perceiver IO adds output queries that read from the latent. Transfer: Perceiver IO's per-output query is the right read pattern for us: each bot queries the shared latent with its own position/map value, rather than receiving the same broadcast. Top-1 ImageNet figure for Perceiver (~76.4%) is from memory only. [from memory]

**PointNet++** — Qi, Yi, Su, Guibas, 2017 (1706.02413). https://arxiv.org/abs/1706.02413 [verified]
Hierarchical grouping: PointNet applied on nested local regions at growing scales, plus multi-scale/multi-resolution grouping for non-uniform density. Original PointNet (global max-pool only, 1612.00593) [from memory] lacked local structure, which PointNet++ added. Transfer: the opposite direction to ours: a pure global pool misses local structure and a pure local misses global; the hierarchy (coarse grid) is the middle path.

**Vision Transformers Need Registers** — Darcet, Oquab, Mairal, Bojanowski, 2023 (2309.16588). https://arxiv.org/abs/2309.16588 [verified]
ViTs spontaneously repurpose low-information tokens as high-norm "global scratch" tokens; adding explicit register tokens removes artifacts and cleans attention maps. Transfer: strong evidence that networks WANT a small global scratchpad and will create one if not given; so giving a few global tokens is low-risk and might improve cleanliness. Also a pitfall: registers may be used as scratch, not as inter-agent communication.

**EPiC-GAN: Equivariant Point Cloud Generation for Particle Jets** — Buhmann, Kasieczka, Thaler, SciPost Phys. 15, 130 (2023) (2301.08128). https://arxiv.org/abs/2301.08128 [verified]
Deep-sets-based layers with a global vector g (dim 10, 128 hidden per point in the generator, 6 layers): g' = MLP(g, [sum-pool, mean-pool](points)), then each point p_i' = MLP(g', p_i). No pairwise edges. The authors state both pooling operations were necessary but report no numbers, and report no ablation on dim(g) or on removing g (ar5iv extraction).
Transfer: the closest architectural precedent for a bandwidth-limited global channel in a generative, permutation-equivariant model, with sum AND mean pooling. The absence of a dim(g) ablation means nobody has quantified this; our experiment would be informative.

**EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion** — Buhmann, Ewen, Faroughy, Golling, Kasieczka, Leigh, Quetant, Raine, Sengupta, Shih, 2023 (2310.00049). https://arxiv.org/abs/2310.00049 [verified]
EPiC layers in score-based diffusion (EPiC-JeDi) and the first permutation-equivariant flow-matching model (EPiC-FM) for jets; claimed faster than transformer diffusion with comparable quality, EPiC-FM best on all metrics. Transfer: shows a global-vector deep-sets backbone is competitive in a flow-matching generative model; it is the architecture to copy for a "global only" baseline.

**TokenFlow: Latent Token Rectified Flow Matching for 3D Point Cloud Generation** — Wang, Huang, 2026 (2610.27418, submitted 20 Oct 2026). https://arxiv.org/abs/2610.27418 [verified abs page]
A small bottleneck of learnable tokens (far fewer than points) gathers global context via cross-attention at O(NM) cost; rectified flow over the tokens, conditional flow over points. Reports best MMD/coverage on ShapeNet. Recent preprint, unreviewed. Transfer: contemporary evidence for latent-token bottlenecks in flow matching; motivates M-token design.

**EMERGE** — Mitsouras, Chaidos, Stamou, Voulodimos, 2026 (2609.26039). https://arxiv.org/abs/2609.26039 [verified abs page; the search snippet said it uses inducing-point attention as a global block, which I did not confirm in text] Unreviewed preprint.

**Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets)** — Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, ICLR 2021 (2010.03409). https://arxiv.org/abs/2010.03409 [verified]; **MultiScale MeshGraphNets** — Fortunato, Pfaff, Wirnsberger, Pritzel, Battaglia, 2022 (2210.00612) [snippet only]
Fine meshes need more propagation steps to cover the same physical distance; a coarse-mesh or two-resolution message passing improves accuracy at lower cost (claimed, not checked). Transfer: direct precedent for the "coarse grid" option: add a coarse graph on top of the fine one, bots scatter onto it and read back.

### 2.3 Multi-agent communication

**Learning Multiagent Communication with Backpropagation (CommNet)** — Sukhbaatar, Szlam, Fergus, NIPS 2016 (1605.07736). https://arxiv.org/abs/1605.07736 [verified]
Each agent's comm input is the mean of the other agents' hidden states (1/(J-1) normalised; restricted to range if local), K=2 comm steps for the MLP module, skip connections. Traffic junction failure (%), 5 runs (ar5iv extraction): Independent 20.6 +- 14.1 (MLP), 19.5 +- 4.5 (RNN), 9.4 +- 5.6 (LSTM); Fully-connected 12.5/34.8/4.8; Discrete comm 15.8/15.2/8.4; CommNet 2.2 +- 0.6 / 7.6 +- 1.4 / 1.6 +- 1.0.
Transfer: the mean message is the most direct precedent for a pooled global vector, and it clearly helped on a task requiring coordination. Note also that fully-connected (naive all-to-all concat) did worse than mean, so the normalised mean is a good inductive bias. Variance across seeds is large for the baseline, so use many seeds.

**TarMAC** — Das, Gervet, Romoff, Batra, Parikh, Rabbat, Pineau, ICML 2019 (1810.11187). https://arxiv.org/abs/1810.11187 [verified]
Traffic junction success (ar5iv extraction): no communication 84.9 +- 4.3 (easy) / 74.1 +- 3.9 (hard); CommNet 99.7 / 78.9 +- 3.4; TarMAC 1-round 99.9 / 84.6 +- 3.2; TarMAC 2-round 99.9 / 97.1 +- 1.6. Ablation (text, exact values only in the Figure 3 plot): two rounds clearly beat one; shrinking the message to a single scalar performs almost as well as 64-d; more rounds helped more than wider messages. Attention vs no attention on SHAPES: 82.4 vs 85.8; House3D 64.3 vs 68.9.
Transfer: (a) very low bandwidth (1 scalar) can carry most of the benefit; (b) iterating (rounds) beats widening; (c) targeted/attention reads beat uniform mean by a modest, task-dependent margin. Hence design: few floats but several read-write rounds, and attention-style read.

**Graph Convolutional Reinforcement Learning (DGN)** — Jiang, Dun, Huang, Lu, ICLR 2020 (1810.09202). https://arxiv.org/abs/1810.09202 [verified]. Adaptive graph convolution with relation kernels over each agent's neighbourhood; I did not check numbers. Transfer: local relational kernels with shifting neighbours is a MARL analogue of radius-graph attention; supports local-first design.

**Learning when to Communicate at Scale (IC3Net)** — Singh, Jain, Sukhbaatar, ICLR 2019 (1812.09755). https://arxiv.org/abs/1812.09755 [verified]. Gated continuous communication, per-agent rewards. A search snippet reported ~30% success with no communication vs 93.0% with a continuous 128-d shared hidden state on traffic-junction easy (from a secondary snippet; unverified). [snippet only for numbers]. Transfer: gating (learned "when to talk") is the natural sparsification; could gate each bot's write to the global state.

**Learning Individually Inferred Communication (I2C)** — Ding, Huang, Lu, NeurIPS 2020 (2006.06455). https://arxiv.org/abs/2006.06455 [verified]. Argues broadcast comms are impractical and redundant; each agent learns whom to talk to via causal influence, plus policy regularisation to exploit messages. Transfer: explicitly regularising the policy to use messages is a counter to "ignored" collapse; causal-influence labels are a measurement tool we can borrow.

**Mean Field Multi-Agent Reinforcement Learning** — Yang, Luo, Li, Zhou, Zhang, Wang, ICML 2018 (1802.05438). https://arxiv.org/abs/1802.05438 [verified]. Approximates the interaction with all other agents by the average action of neighbours (Q depends on own state and mean-neighbour action). Transfer: justifies low-dimensional statistics as sufficient in the large-N limit; note it is a neighbourhood mean (local), not global.

**Scalable Task-Driven Robotic Swarm Control via Collision Avoidance and Learning Mean-Field Control** — Cui, Li, Fabian, Koeppl, ICRA 2023 (2209.07420). https://arxiv.org/abs/2209.07420 [verified]. Mean-field control reframes swarm control as a single-agent problem over distributions; assumes weak interactions, so collision avoidance is added; outperforms MARL in simulation and on UAVs. Transfer: the distribution/histogram state is a theoretically justified state for large swarms.

**Bandwidth-limited comms**: Kapoor, Bhisikar, Freed, Peters, Sun, "Learning what to say and how precisely: Efficient Communication via Differentiable Discrete Communication Learning" (2511.01554, Nov 2025). https://arxiv.org/abs/2511.01554 [verified, preprint]. Learns message precision at bit level; reports >10x bandwidth cut with matched or better performance across four MARL algorithms, and a simple Transformer policy plus this layer comparable to specialised comm architectures (abstract claims). Earlier line (Freed et al. 2020, up to 34x fewer bits; IMAC, Wang et al. ICML 2020, information-bottleneck scheduler) [snippet only; IDs not recorded, do not cite without lookup].
Transfer: bandwidth can usually be cut a lot without loss; so a few floats is a realistic bandwidth. Quantisation noise as a regulariser/also physical realism (a gossip channel is noisy and delayed).

**On the Pitfalls of Measuring Emergent Communication** — Lowe, Foerster, Boureau, Pineau, Dauphin, AAMAS 2019 (1903.05168). https://arxiv.org/abs/1903.05168 [verified]
Distinguishes positive signalling (messages correlate with sender state/action) from positive listening (receiver behaviour changes). Found a case where messages predicted future actions yet "do not impact the environment or other agent in any way". Proposes causal influence of communication (CIC): intervene on messages and measure the shift in the listener's action distribution. A later survey (2006.02419) [snippet only] notes ablating the channel and seeing performance fall is not by itself proof.
Transfer: this is the evidence for "ignored / collapse"; adopt CIC as a required metric.

### 2.4 Swarm / flocking with GNNs and global info

**Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks** — Tolstaya, Gama, Paulos, Pappas, Kumar, Ribeiro, CoRL 2019 (1903.10527; v4 Mar 2021). https://arxiv.org/abs/1903.10527 [verified; venue CoRL 2019 from memory]
Local controllers (only nearby robots, multi-hop GNN) trained by imitating a centralised controller with global information, on flocking (N=100, v_init up to 3 m/s). Text findings (ar5iv extraction; numbers only in figures): K=1 is an order of magnitude worse than K=2-4; at R=1.0 m, K=3-4 GNNs are much better than all but the global controller; at large radii (R=4 m) the purely local controller's cost goes below the GNNs. Abstract: multi-hop information is more valuable as radius shrinks and velocity rises.
Transfer: (a) the global controller is the upper bound at small radius; (b) multi-hop depth is a substitute for a global channel, so a fair test equates "receptive field" budgets; (c) at large radius extra machinery can hurt, so expect a regime where the global state adds nothing.

**Meanshift Shape Formation Control Using Discrete Mass Distribution** — Cai, Gao, Li, Wang, Sun, Lu, 2026 (2602.00980). https://arxiv.org/abs/2602.00980 [verified abs page; preprint]
Fully decentralised shape formation with local communication only: each robot estimates masses of the swarm over a discrete set of sample points via a mass estimator (the estimates converge asymptotically to global values), and a meanshift law matches the swarm distribution to the target distribution; handles swarm-size changes. Not learned.
Transfer: the best physical template for our shared state: an M-float vector (mass per target-sample cell) obtained by consensus, which is a spatial histogram (the "coarse grid") with explicit bandwidth M. Includes a convergence argument for consensus-estimated global quantities.

### 2.5 Pitfall evidence on explicit global states in generative models
I found no paper that reports an ablation in which a global token/virtual node in a permutation-equivariant diffusion/flow model was shown ignored (search for this came back empty; the EPiC papers have no ablation). This is a gap, not evidence in either direction. [search result negative]

## 3. What to expect (inference, not from the papers)

- **Pooled vector (mean+sum, ~8-32 floats) added to the radius-r local denoiser:** likely to recover part of the small-r gap on tasks needing centroid/size/progress (e.g. getting the pile centred, global scale), and to relieve the need for a clock t if it can infer progress from density statistics. Likely not enough for unseen shapes with holes/fragments, because "which region needs particles" is a map, not a few moments. Prediction: improvement at r ~ 2-3 but still behind r ~ 6 on thin/fragment shapes.
- **Coarse grid scatter/read (e.g. 8x8 or 16x16 cells x C channels):** the one most likely to match global attention on holes and fragments, since it carries spatially indexed information; bandwidth is G^2 x C floats, so compare to pooled vector at matched floats and matched params. TarMAC and the Southern et al. VN theory suggest the benefit comes from position-dependent reads.
- **Few inducing tokens with position-dependent attention reads (Perceiver IO style):** between the two; M of 4-16 probably saturates; expect diminishing returns beyond that (TarMAC: 1 scalar ~ 64-d on its task; but our task is higher-entropy).
- **Multi-hop depth at r small vs global channel:** Tolstaya suggests depth alone partly compensates; expect L x r to explain much of the r=6 result. If global state wins at equal receptive field, that is the interesting result.
- **Unseen shapes:** global vectors learned on training shapes risk encoding shape identity (memorisation) rather than reusable statistics; expect generalisation gap unless the state is a function of the swarm and map only through pooling of per-bot features.
- **Collapse risk:** with a strong local branch and a map given at each bot, the global channel may carry only time-like info; registers evidence (ViT) and positive-signalling-without-listening suggest it may be used as scratch.
- **Noise:** consensus gives delayed/noisy estimates; if trained noiseless, deployment may degrade. Add noise/delay augmentation if physical realisability is a claim.

## 4. Pitfalls (how not to fool ourselves)

1. **Intervene, don't just ablate (Lowe et al. CIC).** At test time replace the global state with (a) zeros, (b) the state from a different sample (batch shuffle), (c) the state from a different time step. Report the shift in the denoiser output / final shape metric. If a shuffle changes nothing, it is not used.
2. **Shuffle controls must keep statistics.** Shuffling across batch items of the same shape and t separates "uses this swarm's info" from "uses a constant bias or t". Also compare against a learned constant (same shape, no input) of the same size.
3. **Match bandwidth.** Compare variants at equal floats-per-step (pooled D floats vs M tokens x d vs G^2 x C cells), and report it explicitly; say how many reads/writes per step (rounds), since TarMAC found rounds matter more than width.
4. **Match parameters and compute.** Global branch adds MLP params and FLOPs; add the same to the local baseline (wider/deeper, or K more hops) and to a null global channel (same module but fed a constant). Equalise receptive field L x r too.
5. **Clock confound.** Our local denoiser needs t; a global state can trivially encode t or progress. Run with and without t, and with a global state that cannot see positions (only a learned constant), to see which gain is clock.
6. **Seeds and variance.** CommNet baselines had std up to 14 points on 5 runs; use >=5 seeds and report intervals; evaluate on held-out thin/hole/fragment shapes and on N != 256 if scale-invariance is claimed.
7. **Pooled-vector capacity.** Per Wagstaff et al., the width needed for exact function representation grows with set size, so a null result at D=8 does not imply the idea is bad; sweep D (and use sum+mean, EPiC style) before concluding.
8. **Information leak.** If the map is read by each bot at its position, global state computed from the map values of all bots is an oracle on the target shape beyond what physical consensus could supply in finite time/hops; model the gossip delay or limit the number of consensus rounds per step, otherwise "physically realisable" is not earned.
9. **Registers/scratch confusion.** A tokens-as-scratch gain (as in ViT registers) is not inter-bot communication; test by making the tokens per-bot (no cross-bot flow) with same parameters.
10. **Citation hygiene.** Items tagged [snippet only] or [from memory] above (Gilmer virtual node claim, Perceiver 76.4%, IC3Net 30%/93% figures, Local VN / RANGE / traffic VN papers, Freed 2020 and IMAC IDs, survey 2006.02419) should be re-verified before external use. Numbers marked "ar5iv extraction" were summarised by a fetch tool, so check the paper table.
