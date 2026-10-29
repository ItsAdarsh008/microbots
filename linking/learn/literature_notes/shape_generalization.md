# Procedural target-shape generation and generalization to held-out shape families in learned self-assembly / reconfiguration / collective construction

Scope note: about 20 searches and fetches. The direct literature on *learned, lattice-based, connectivity-constrained swarm assembly with held-out shape families* is very thin. Most evidence is adjacent: single-builder block assembly, Minecraft/NCA growth, UED/PCG for RL in general, and the purely algorithmic reconfiguration theory. Items are labeled **[2024+]** or **[background]**.

## Q1. Which works train learned policies to build or morph into arbitrary target shapes, how are targets generated, and how is generalization measured?

### Takeaway
The best quantitative evidence on shape generalization is still Blocks Assemble! (ICML 2022, background). It trained one policy on 141 hand-designed blueprints and tested on 24 held-out ones. Generalization worked only with multi-blueprint training and a graph/relational (object-centric attention) policy: held-out success was 89.6% on large structures and 77.5% on "special" out-of-family structures, versus 0% for a ResNet policy. Work from 2024–2026 (TeamCraft, "Learning to Build" 2026, conditional NCAs) generally reports that policies interpolate within the training distribution and degrade on structurally different layouts. Almost no learned lattice self-reconfiguration paper reports held-out *shape-family* tests.

### Cited Findings
**Block assembly with RL (background, most quantitative)**
- Ghasemipour, Freeman, David, Gu, Kataoka, Mordatch, "Blocks Assemble! Learning to Assemble with Large-Scale Structured Reinforcement Learning", ICML 2022 (PMLR v162). Blueprints were **hand-designed, not procedural**: "we designed 165 blueprints (split into 141 train, 24 test)", 2–16 blocks. The authors note procedural generation is possible but did not use it. The test set holds 12–16-block structures plus special "literal, tallest, longest" structures. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2); [PMLR](https://proceedings.mlr.press/v162/ghasemipour22a/ghasemipour22a.pdf)
- Blocks Assemble success rates at 1B steps. Train: 99.9% (2–6 blocks), 97.9% (7–11), 87.3% (12–16). Held-out: 89.6% (12–16 blocks), **77.5% on held-out "special" structures**, which is the closest analogue to a held-out family. Held-out large blueprints scored about the same as the training ones; the out-of-family set scored about 10–12 points lower. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Blocks Assemble, multi- vs single-blueprint training. A 16-block blueprint scored **0% when trained alone vs 90.9% inside multi-blueprint training**. Single-task agents "mostly failed to transfer" to blueprints they were not trained on, so task diversity acts as a curriculum and a regularizer. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Blocks Assemble, architecture ablation (held-out 12–16 / held-out special):
  - default graph attention: 89.6% / 77.5%
  - no graph attention: 3.6% / 4.8%
  - single gripper: 55.4% / 42.8%
  - non-graph ResNet: 0% / 0%, and never above 2.5% on any set
  
  Relational structure is the dominant factor for shape generalization. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Blocks Assemble, curriculum. Sampling was weighted toward blueprints with lower recent success (Appendix D). The authors say the curriculum does "not have a clear-cut benefit, but may be leading to improvements in generalization". — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Brick-by-Brick (Chung et al., NeurIPS 2021, background). An RL builder assembled LEGO-like bricks to match target images and beat a supervised baseline, which "does not generalize to unseen images of the test dataset". — [arXiv 2110.15481](https://arxiv.org/pdf/2110.15481)
- **[2024+/2026]** "Learning to Build: Autonomous Robotic Assembly of Stable Structures Without Predefined Plans" (arXiv 2602.23934, 2026). A goal-conditioned assembly policy "generalizes reasonably well to spatial shifts and local geometric variations, but is less reliable for unseen layouts that differ more substantially from the training tasks". — [arXiv 2602.23934](https://arxiv.org/html/2602.23934)
- **[2026]** "Learning Compositional Symbolic Task Rules from Demonstrations with Inductive Logic Programming" (arXiv 2605.26828). Symbolic rules learned from demonstrations in a synthetic block-assembly domain give "strong generalization to harder, held-out tasks with unseen objects". Symbolic or relational abstractions are the counterpoint to pure neural nets. — [arXiv 2605.26828](https://arxiv.org/html/2605.26828v1)
- Background: target-stacking with goal-parameterized deep RL (Li et al., 2017) and Neural Task Programming (Xu et al., 2017; 2000 procedurally distinct block-stacking instances with held-out test cases). — [arXiv 1711.00267](https://arxiv.org/abs/1711.00267); [arXiv 1710.01813](https://arxiv.org/pdf/1710.01813)

**Collective / multi-robot construction**
- Background: Sartoretti, Wu, Paivine, Kumar, Koenig, Choset, "Distributed Reinforcement Learning for Multi-Robot Decentralized Collective Construction", DARS 2018. In a TERMES-inspired task, robots gather, carry and place blocks to build a user-specified 3D structure. They need scaffolding to reach higher levels, which gives sparse, delayed rewards. A shared A3C policy is trained centrally and executed decentrally under full observability. The abstract says swarms "of various sizes" built "different test structures" without additional training. The snippets do not say whether those structures were outside the training distribution. — [DARS 2018 PDF](https://idm-lab.org/bib/abstracts/papers/dars18.pdf)
- **[2024+]** TeamCraft (arXiv 2412.05255, Dec 2024) is a multi-agent Minecraft benchmark. Agents build from orthographic blueprints, and held-out splits cover novel goals, scenes and agent counts (train on 2–3 agents, test on 4). The authors report that current models "struggle" to generalize to new objectives, scenes and group sizes. The baselines are VLM-based; the relevant lesson is the evaluation protocol, not the method. — [TeamCraft arXiv 2412.05255](https://arxiv.org/html/2412.05255.pdf)
- Background: Creative Agents (arXiv 2312.02519) uses voxel blueprints for Minecraft building, generated by a Pix2Vox++-style module trained on CraftAssist constructions, and is imitation-based. — [arXiv 2312.02519](https://arxiv.org/pdf/2312.02519)

**Neural cellular automata (NCA)**
- Background: Sudhakaran et al., "Growing 3D Artefacts and Functional Machines with Neural Cellular Automata", ALIFE 2021. A 3D-conv NCA grows Minecraft castles, apartment blocks and trees, some over 3,000 blocks, and regrows parts of simple functional machines. **One model per target structure** (targets taken from Planet Minecraft, about 50 block types), so there is no cross-shape generalization. — [arXiv 2103.08737](https://arxiv.org/pdf/2103.08737); [The Batch summary](https://www.deeplearning.ai/the-batch/3d-object-factory)
- Background: Goal-Guided NCA (arXiv 2205.06806, 2022) adds a goal encoder that adds updates to each living cell's hidden state at every step. It "can generalize its behavior to unseen scenarios in some cases". Its experiments cover image morphing and locomotion, not voxel shapes. — [arXiv 2205.06806](https://arxiv.org/pdf/2205.06806)
- **[2024+]** "Conditional Morphogenesis" (Sakour, arXiv 2512.08360, Dec 2025). One shared NCA rule grows all 10 MNIST digits from a single pixel, conditioned by a one-hot class injected into each cell's perception field (28×28 grid, 16 channels). Only the 10 trained classes were evaluated; there is no held-out shape test. — [arXiv 2512.08360](https://arxiv.org/html/2512.08360)
- **[2024+]** A 2026 NCA review and reference implementation shows that a class-conditioned NCA can generate multiple emoji classes and transition between them. — [arXiv 2604.24990](https://arxiv.org/html/2604.24990)
- **[2024+]** Mesh NCA was trained only on an icosphere mesh yet synthesizes textures on unseen meshes. This shows a local rule generalizing across *substrate geometry*, though not across target shapes. — [arXiv 2311.02820](https://arxiv.org/html/2311.02820)
- **[2024+]** "Smart cellular bricks for decentralized shape classification and damage recovery" (Nature Communications 2026; arXiv 2509.18659). A volumetric 3D NCA on voxel grids classifies shapes and recovers from damage through iterative local state evolution. This is classification, not goal-directed construction. — [Nature Comms](https://www.nature.com/articles/s41467-026-75166-7); [arXiv 2509.18659](https://arxiv.org/html/2509.18659v1)

**Modular / self-reconfigurable robots**
- **[2024+]** Mao & Yao, "Parallel Self-assembly for Modular Robots Using Deep Reinforcement Learning" (ICIRA 2024, LNCS 2025). A separate mapping stage assigns modules to target positions, and SAC handles motion. It is claimed to transfer across robot counts and target configurations (Webots plus experiments). Held-out shape-family numbers were not visible in the abstract. — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0798-3_20)
- Background/2023: "Learning to Self-Reconfigure for Freeform Modular Robots via Altruism MARL" (AAMAS 2023). MARL with a conflict-reducing collaboration mechanism; it argues discrete-primitive planners are infeasible for freeform modules. — [ACM DL](https://dl.acm.org/doi/10.5555/3545946.3598996)
- **[2024+]** Decentralized RL for modular reconfigurable lunar robots (arXiv 2510.20347). Each module has its own policy over local observations. Field tests reported zero-shot integration for locomotion and steering, and "preliminary alignment" for reconfiguration. — [arXiv 2510.20347](https://arxiv.org/html/2510.20347v1)
- Background: Pathak, Lu, Darrell, Efros, "Learning to Control Self-Assembling Morphologies" (NeurIPS 2019). One shared modular policy runs on every limb and generalizes to unseen numbers of limbs and environments. It is often cited as the template for "one shared small net per module". — [paper PDF](https://pathak22.github.io/modular-assemblies/resources/assemblies.pdf)
- **[2024+]** Non-learned shape assembly baseline: mean-shift exploration for robot-swarm shape assembly (Nature Communications 2023). A distributed algorithm builds user-specified shapes; the authors note earlier reaction–diffusion methods cannot make user-specified shapes precisely. — [Nature Comms 2023](https://www.nature.com/articles/s41467-023-39251-5)

### Inferences
- The project's 2 train shapes / 1 test shape setup sits in the Blocks Assemble "single-blueprint" regime, where transfer mostly fails. Its success on the hollow tower is probably helped by the strong local inductive bias (5×5×5 view plus a shared MLP), but one test shape is not enough evidence. Blocks Assemble suggests that about 10² shapes give large gains, and that a *per-family* held-out test (their "special" set) shows a gap of roughly 10 points even when within-family held-out matches training.
- Relational / attention structure over entities (blocks, neighbours) was the decisive factor in Blocks Assemble. For a lattice swarm, the analogue is a translation-equivariant local rule (NCA-like) plus a relational encoding of neighbours and target cells. The current local-view MLP already has this property spatially.
- NCAs show that one local rule *can* produce many shapes when conditioned. The NCA literature has almost no held-out-shape evaluation, so it supports feasibility but not generalization.

### Gaps
- No paper found that trains a learned policy for *connectivity-constrained lattice self-reconfiguration (sliding/pivoting cubes)* on procedurally generated targets and reports held-out-family results. This looks like a genuine open niche.
- Could not confirm whether Sartoretti et al. 2018 tested on structures outside its training distribution, or get its numbers.
- Could not access full texts of Mao & Yao (ICIRA 2024) or the 2026 "Learning to Build" paper to get quantitative held-out numbers.

## Q2. What do PCG, domain randomization, curriculum and UED say about how many and how diverse training tasks are needed for OOD generalization?

### Takeaway
Procgen (2020) set the rule of thumb: generalization gaps close only with thousands to tens of thousands of procedurally generated levels (about 10,000 for full closure; 200 is a cheap "easy" setting). Work from 2024–2025 (Kinetix; No Regrets/SFL) shows two things. First, very large random task spaces plus one general policy give zero-shot transfer to hand-designed tasks. Second, plain domain randomization is a strong baseline that regret-based UED often fails to beat. Sampling tasks of intermediate difficulty ("learnability") helps most.

### Cited Findings
- Background: Cobbe, Hesse, Hilton, Schulman, "Leveraging Procedural Generation to Benchmark Reinforcement Learning" (Procgen), ICML 2020. Training-level pools from 100 to 100,000 showed heavy overfitting with small pools. **As many as 10,000 levels** were needed to close the generalization gap. A **200-level** training set is suggested for the easy-difficulty protocol. Past a threshold, *training* performance also improves as the level set grows, unlike typical supervised trends. — [Procgen PMLR](https://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf); [OpenAI PDF](https://cdn.openai.com/procgen.pdf)
- Background: Justesen et al., "Illuminating Generalization in Deep RL through Procedural Level Generation" (AAAI-19 RLG workshop). Procedurally varied training levels give more general policies than one fixed level, and difficulty-adaptive generation helps. — [paper](https://rlg.mlanctot.info/papers/AAAI19-RLG-Paper22.pdf)
- Background: House3D (Wu et al., 2018). Training on 200 houses vs 20 was harder to fit but generalized better to 50 held-out houses. — [arXiv 1801.02209](https://arxiv.org/pdf/1801.02209)
- **[2024+]** Matthews, Beukman, Lu, Foerster, "Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks", ICLR 2025 (oral). One policy trained on **millions** of randomly generated 2D physics scenes (Jax2D, billions of steps) zero-shot solves unseen hand-designed levels. Fine-tuning it beats training from scratch, and it succeeds on some tasks where standard RL fails. The repository notes performance drops on harder scenarios. — [arXiv 2410.23208](https://arxiv.org/abs/2410.23208v2); [ICLR 2025](https://iclr.cc/virtual/2025/oral/31729); [GitHub](https://github.com/FlairOx/Kinetix)
- **[2024+]** Rutherford, Beukman, Willi, Lacerda, Hawes, Foerster, "No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery", NeurIPS 2024. The regret approximations used in practice (PLR/ACCEL-style) track *success rate*, not regret, so they over-sample already-solved tasks. The proposed **Sampling For Learnability (SFL)** prioritizes levels with success strictly between 0 and 1 and beats UED baselines in Minigrid and a robotics-inspired domain. In the robotics domain, **no SOTA UED method beat plain domain randomization**. The paper also adds a CVaR-like adversarial robustness evaluation. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d0ed12c3fda52f2c241a0cebcf739a6-Abstract.html); [arXiv 2408.15099](https://arxiv.org/abs/2408.15099v3); [talk](https://slideslive.com/39024959)
- Background: DIPLR (IJCAI 2023) argues that environment *diversity*, not just difficulty, is the main lever for OOD performance in UED. It uses a teacher that generates diverse levels. — [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0601.pdf)
- **[2026]** PACE (arXiv 2605.01358). Its domain-randomization baseline in MiniGrid builds each level from 100 random obstacle placements and is evaluated zero-shot on 12 human-designed levels. This is a representative modern UED protocol: train on random levels, test on hand-designed OOD levels. — [arXiv 2605.01358](https://arxiv.org/html/2605.01358)
- Background (the standard UED lineage, all cited by the works above):
  - PAIRED (Dennis et al., NeurIPS 2020): adversary-generated levels via protagonist/antagonist regret.
  - Prioritized Level Replay (Jiang et al., ICML 2021): replay levels with high TD-error/value-loss scores.
  - ACCEL (Parker-Holder et al., ICML 2022): evolve levels by small edits to high-regret replayed levels.
  
  — [PAIRED arXiv 2012.02096](https://arxiv.org/abs/2012.02096); [PLR arXiv 2010.03934](https://arxiv.org/abs/2010.03934); [ACCEL arXiv 2203.01302](https://arxiv.org/abs/2203.01302)
- In Blocks Assemble (Q1), an inverse-success sampling curriculum gave no clear benefit, but multi-task diversity was critical: 0% → 90.9% on 16-block blueprints. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)

### Inferences
- For "thousands of random buildable shapes", the literature supports going to at least 10³–10⁴ distinct targets. Expect training performance itself to *improve* with more shapes (Procgen), and expect DR (uniform sampling from the generator) to be a hard baseline to beat.
- The cheapest high-value curriculum is SFL-style: keep shapes where the swarm sometimes but not always completes, judged on per-shape success. With DAgger, this maps to sampling shapes where the student most often diverges from the teacher. ACCEL-style "mutate a shape by adding or removing a few voxels" is a natural fit for voxel targets.
- Held-out *families* (for example "all shapes with internal cavities" or "all arches/overhangs") match the Kinetix/PACE protocol of random training plus hand-designed OOD tests. Report both within-family held-out and cross-family held-out, as Blocks Assemble did with its "special" set, because the two can differ by 10+ points.

### Gaps
- No source quantifies "how many shapes" specifically for construction or assembly policies. Procgen numbers are for game levels and may not transfer.
- No 2024+ UED paper found that applies UED to shape-conditioned construction or reconfiguration.

## Q3. What is known theoretically about which shapes are buildable/reachable (sliding/pivoting cubes, TERMES), and how is a "buildable class" defined?

### Takeaway
For **sliding cubes** that must stay face-connected, reconfiguration is *universal*: any connected n-module shape can reach any other in O(n²) moves, which is tight (Abel et al., SoCG 2024). In-place, input-sensitive and parallel refinements followed in 2024–2025. For **pivoting cubes** (the M-Blocks-style move, which needs free sweep space), universality *fails* in general. Positive results need extra "helper" modules (2D: O(1) musketeers) or exclude certain sub-configurations (3D, Sung et al. 2015). Related 3D pivoting models are PSPACE-hard and contain rigid configurations. So with pivoting the "buildable class" must be defined by the move model, not just by connectivity.

### Cited Findings
- **[2024+]** Abel, Akitaya, Kominers, Korman, Stock, "A Universal In-Place Reconfiguration Algorithm for Sliding Cube-Shaped Robots in a Quadratic Number of Moves", SoCG 2024 (LIPIcs 293). This is the first universal algorithm for 3D sliding cubes:
  - It converts any face-connected n-module configuration to any other using only slide moves.
  - It uses **O(n²) moves, asymptotically tight**, since some pairs need Ω(n²).
  - Face-connectivity is maintained at every step.
  - The in-place variant keeps all modules except one within the union of the start and target bounding boxes.
  - It extends and subsumes Abel & Kominers' 2008 arXiv preprint "Universal reconfiguration of (hyper-)cubic robots", and cites Dumitrescu & Pach, "Pushing squares around" (Graphs and Combinatorics 2006) as the 2D predecessor.
  
  — [Dagstuhl LIPIcs SoCG 2024](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SoCG.2024.1)
- **[2024+]** Kostitsyna et al., "Optimal In-Place Compaction of Sliding Cubes" (SWAT 2024). Reconfiguring into a compact shape takes moves equal to the sum of input coordinates, which ranges from Ω(n^{4/3}) to O(n²). — [arXiv 2312.15096](https://arxiv.org/pdf/2312.15096); [Dagstuhl SWAT 2024](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SWAT.2024.31); summarized in [arXiv 2507.04170](https://arxiv.org/abs/2507.04170)
- **[2024+]** Akitaya, Korman, Stock, "Input-Sensitive Reconfiguration of Sliding Cubes" (CCCG 2025; arXiv 2507.04170, July 2025). Universal and input-sensitive: the move count depends on the coordinate sums of both start and target. Computation is O(1) amortized per move, and face-connectivity is maintained. — [arXiv 2507.04170](https://arxiv.org/abs/2507.04170)
- **[2024+]** Akitaya et al., "Sliding Squares in Parallel" (ESA 2025; arXiv 2412.05523, Dec 2024). With simultaneous moves, the worst-case optimal makespan is O(P), where P is the bounding-box perimeter. Deciding whether one parallel step suffices is NP-complete in the unlabeled case. There is a feasible, weakly in-place schedule of length linear in both bounding-box perimeters. — [arXiv 2412.05523](https://arxiv.org/html/2412.05523); [LIPIcs ESA 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.28)
- **[2024+]** Distributed sliding-square algorithms (2025): "Distributed Diamond/Rhombus Formation of Sliding Squares" gives a sequential O(n²)-round method and parallel variants using a spanning tree. These are local, distributed rules, the closest theory analogue to a learned shared policy. — [Springer invited paper](https://link.springer.com/chapter/10.1007/978-3-032-11127-2_26); [ResearchGate preprint](https://www.researchgate.net/publication/394473224_Distributed_Diamond_Formation_of_Sliding_Squares)
- Background: Akitaya, Arkin, Damian, Demaine, Dujmović, Flatland, Korman, Palop, Parada, van Renssen, Sacristán, "Universal Reconfiguration of Facet-Connected Modular Robots by Pivots: The O(1) Musketeers" (ESA 2019; Algorithmica 2021). 2D square-lattice pivoting is universal **given 5 extra helper modules**, using O(n²) pivot moves (worst-case optimal). Without helpers, pivoting squares are not universal. — [arXiv 1908.07880](https://arxiv.org/html/1908.07880)
- Background (2023): "Reconfiguration of 3D Pivoting Modular Robots" (rhombic dodecahedral modules). Reconfiguration is **PSPACE-hard** in a restricted pivoting model. Under a broader move set, configurations cannot always reach each other, unlike the 2D analogue. Some configurations stay **rigid** even as a subset of any larger configuration. — [arXiv 2304.09990](https://arxiv.org/html/2304.09990)
- Background: the ElectroVoxel paper (2022) summarizes Sung, Bern, Romanishin, Rus (ICRA 2015) as saying existing 3D pivoting-cube planners handle arbitrary lattice shapes in O(n²) moves, **except for three excluded sub-configurations**. It also validated the pivot and traversal primitives in hardware. — [ElectroVoxel arXiv 2207.05732](https://arxiv.org/html/2207.05732)
- Background, TERMES: Werfel, Petersen, Nagpal, "Designing Collective Behavior in a Termite-Inspired Robot Construction Team", Science 343:754–758 (2014). A user specifies a desired structure and the system automatically generates low-level rules that guarantee it gets built. Robots use only local sensing and coordinate through stigmergy (the shared environment). The method was demonstrated with 3 climbing robots building towers, castles and pyramids, adding stairs as needed. Earlier version: Petersen, Nagpal, Werfel, RSS 2011. — [Harvard SSR page](https://del.seas.harvard.edu/publication/designing-collective-behavior-termite-inspired-robot-construction-team); [SEAS news](https://seas.harvard.edu/news/2014/02/robotic-construction-crew-needs-no-foreman)

### Inferences
- If the simulated bots can always *slide* (including convex transitions over edges, as in the Abel et al. model), then every face-connected target of the right voxel count is reachable. The constraint "stay connected to the floor" is stricter than plain face-connectivity of the module set, but it is satisfied if the floor is treated as a fixed, connected anchor module set. The buildable class is then essentially "face-connected voxel sets that touch the floor", and a generator can sample freely from it, for example by random connected growth from floor seeds.
- If only *pivoting* is available (true M-Blocks), some shapes or sub-configurations may be unreachable or rigid. The generator should then filter targets by running the hand-written teacher, or a planner, and keep only those it completes: "buildable = teacher-solvable". This is the pragmatic definition used implicitly by imitation-learning pipelines. It also means held-out families must be checked for teacher-solvability, or failures will conflate policy error with infeasibility.
- Theory gives O(n²) sequential and O(perimeter) parallel move bounds. These are natural normalizers for reporting learned-policy efficiency on random shapes, for example the ratio of moves used to n² or to the bounding-box perimeter.

### Gaps
- Could not retrieve TERMES' exact buildability conditions (the "structpath" compiler constraints on height steps and gaps between adjacent bricks) from a primary source. I recall these from memory but did not verify them, so they are left out of the findings.
- Did not find a 2024–2026 universal-reconfiguration result for *3D pivoting cubes* specifically. The Dumitrescu & Pach 2006 primary text was not fetched; it is cited only via SoCG 2024's reference list.
- No source formally defines a "buildable class" for learned-policy training sets. Learned works (Blocks Assemble) used hand-designed blueprints instead.

## Q4. Goal- and shape-conditioned policies: which representations of the target generalize best (local slices vs global encoding vs SDF)?

### Takeaway
No head-to-head comparison of target encodings (local window vs global latent vs SDF) for construction policies was found. The indirect evidence is fairly consistent: **object- or cell-centric, relational, local encodings** (graph attention over blocks; per-cell conditioning in NCAs; local rules that transfer across meshes) generalize, while global CNN/ResNet encodings of the whole scene fail (0% in Blocks Assemble). Class or one-hot global codes (conditional NCAs) do not, by construction, extrapolate to new shapes.

### Cited Findings
- Blocks Assemble, structured vs unstructured. Graph attention over entities reached 89.6% held-out success; dropping graph attention gave 3.6%; a ResNet over the whole scene gave 0%. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Goal-Guided NCA injects a goal encoding additively into every cell's hidden state at every step. It "can generalize ... to unseen scenarios in some cases". — [arXiv 2205.06806](https://arxiv.org/pdf/2205.06806)
- Conditional NCA (2025) injects a one-hot class into each cell's perception and was only evaluated on its 10 trained classes. A categorical code gives no route to novel shapes. — [arXiv 2512.08360](https://arxiv.org/html/2512.08360)
- Mesh NCA: a purely local rule trained on one mesh transfers to unseen meshes, which shows that local-only perception yields geometry transfer. — [arXiv 2311.02820](https://arxiv.org/html/2311.02820)
- **[2024+]** MeshGraphNets on unseen obstacle geometries (fluids, 2024): GNNs "can sometimes generalize well to various shapes", but not reliably. This is a caution against assuming GNN locality guarantees shape transfer. — [arXiv 2408.06101](https://arxiv.org/html/2408.06101v1)
- **[2024+]** Mao & Yao (ICIRA 2024) separate *where each module should go* (an explicit mapping/assignment stage) from *how to move* (learned SAC). This decoupling makes the learned part independent of global shape. — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0798-3_20)
- Background: Pathak et al. 2019. A shared per-module policy with message passing generalizes to unseen module counts. — [paper](https://pathak22.github.io/modular-assemblies/resources/assemblies.pdf)
- Background: TERMES compiles the global target offline into local rules or a path, so robots only read local target information at runtime. — [Harvard SSR page](https://del.seas.harvard.edu/publication/designing-collective-behavior-termite-inspired-robot-construction-team)

### Inferences
- The project's design already follows the "local slice of target + relayed scalar field" pattern: a local target map, a hop-count gradient (like a morphogen or TERMES-style compiled path), and a compass. The evidence suggests this is the right family. The global component should be a *field computed from the target* (distance-to-unfilled-target, hop count, height or ordering layer) rather than a global latent code of the shape.
- An SDF-like or distance-transform channel (distance to the nearest unfilled target voxel, or to the target surface) is a natural generalization of the hop-count signal. It is shape-agnostic by construction. No paper was found that directly tests SDF conditioning for construction policies, so this is a hypothesis to test.
- Ordering/deadlock information (which voxel to fill next) depends on global shape. The safest transferable route is to compute it as a local field (for example a layer index or "dependency depth" relayed like the hop count) rather than having the MLP infer it from a 5×5×5 window.

### Gaps
- No paper directly compares local-window, global-embedding and SDF target encodings for voxel construction or reconfiguration generalization. This is a clear gap and a possible contribution.

## Q5. Known failure modes when generalizing to new shapes (enclosed cavities, overhangs, ordering deadlocks)

### Takeaway
Reported failure modes are larger or longer structures, layouts structurally unlike the training set, and scaffolding/ordering dependencies. Theory adds rigid or locked configurations and unreachable sub-configurations under pivoting. Explicit studies of cavities and overhangs in *learned* policies were not found. The evidence comes from performance drops on "special" structures, from the theory of locked and rigid states, and from the scaffolding requirement in TERMES-like construction.

### Cited Findings
- Size or length: Blocks Assemble success falls from 99.9% (2–6 blocks) to 87.3% (12–16 blocks) even on training blueprints, and to 77.5% on held-out "special" (tallest/longest/literal) structures. Without blueprint resets, reset-free success on 12+ blocks drops to 69.4%. — [arXiv HTML v2](https://arxiv.org/html/2203.13733v2)
- Structural dissimilarity: a 2026 goal-conditioned assembly policy is "less reliable for unseen layouts that differ more substantially from the training tasks". — [arXiv 2602.23934](https://arxiv.org/html/2602.23934)
- Harder scenarios: Kinetix's general agent degrades on harder hand-designed levels. — [GitHub](https://github.com/FlairOx/Kinetix)
- Sparse, delayed reward and scaffolding: in TERMES-like RL construction, robots must build scaffolding to reach upper levels, which makes the problem "a very difficult game with sparse and delayed rewards". — [Sartoretti et al. DARS 2018](https://idm-lab.org/bib/abstracts/papers/dars18.pdf)
- Rigid or locked states: in 3D pivoting-module models, some configurations are rigid even inside any larger configuration, and reconfiguration is PSPACE-hard in restricted models. — [arXiv 2304.09990](https://arxiv.org/html/2304.09990)
- Excluded sub-configurations: 3D pivoting-cube planners (Sung et al. 2015, as summarized) exclude three sub-configurations. — [arXiv 2207.05732](https://arxiv.org/html/2207.05732)
- 2D pivoting is not universal without helper modules (hence the 5 "musketeers"). — [arXiv 1908.07880](https://arxiv.org/html/1908.07880)
- Parallel scheduling: deciding whether one parallel step suffices is NP-complete (sliding squares). This hints that conflict-free parallel ordering is intrinsically hard for decentralized policies. — [arXiv 2412.05523](https://arxiv.org/html/2412.05523)
- Swarm multi-agent conflicts: MARL self-reconfiguration work adds explicit collaboration or altruism mechanisms because naive independent policies produce conflicts. — [ACM DL AAMAS 2023](https://dl.acm.org/doi/10.5555/3545946.3598996)
- Multi-agent construction benchmark: models struggle with novel goals and larger agent counts (trained on 2–3, tested on 4). — [TeamCraft](https://arxiv.org/html/2412.05255.pdf)

### Inferences
- For the cube swarm, the most likely held-out-family failures are:
  - **Enclosed cavities / hollow shapes.** Bots must not seal the interior before it is vacated, and in sliding/pivoting models bots inside a cavity may be trapped.
  - **Overhangs and arches.** These need temporary support (scaffolding) whose removal must preserve floor connectivity. The project's "holder tree" local fall check targets this.
  - **Ordering deadlocks.** Filling a voxel that blocks access paths to unfilled target voxels, or the parallel conflicts implied by the NP-completeness result.
  - **Narrow features** (one-voxel-wide pillars or slots). Pivoting needs free sweep space.
- Recommended held-out families that isolate these mechanisms: "contains an enclosed cavity", "has overhang depth ≥ k", "has 1-wide slots", "height/size beyond the training range", "multiple disconnected-from-each-other but floor-connected towers". Report each separately, following the Blocks Assemble "special set" idea.
- Teacher-solvability filtering is essential. Any target the hand-written teacher cannot finish should be excluded or reported separately, otherwise DAgger labels will be degenerate on exactly the hard families.

### Gaps
- No learned-policy paper found that specifically analyzes cavities, overhangs or deadlocks as generalization failure modes in lattice construction. The connections above are inferences from theory and from adjacent results.
- No quantitative data found on how much scaffolding-requiring targets degrade learned construction policies.
