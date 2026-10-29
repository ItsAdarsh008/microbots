# RL Under Physical Constraints for Modular Robots, Robotic Assembly and Collective Construction

Scope note: research done 2026-10-28 with about 23 search/fetch calls. Items marked **[Background]** are pre-2024. Items marked **[from prior knowledge, URL not fetched this session]** are well-known papers whose details I did not re-check in this session. The report writer should treat those as lower confidence. Everything else was checked against an abstract or full-text page fetched this session.

Project context used to judge relevance: cube robots on a 3D lattice (slide/pivot), must stay face-connected to the floor, with a hard local "holder tree" safety mask. Connectors are weak (~3 N, about one bot of sideways cantilever). Bridges currently need a hand-written support-pillar planner. One arch shape deadlocks. Plan: imitation learning (IL), then RL fine-tuning under the real strength limit, with the mask kept on.

---

## 1. Action masking and shielding: does masking hurt exploration or learning, and how to combine masks with imitation-pretrained policies?

### Takeaway
Most work agrees that hard masking of invalid or unsafe actions helps policy-gradient learning and does not hurt it. The theory (2020) and newer analysis (2026) show that *not* masking is what damages learning: invalid-action gradients leak through shared weights and suppress valid actions in states the agent has not visited yet. The main known pitfall is a train/deploy mismatch. A policy trained with a mask depends on it and fails if the mask changes or is removed. For an IL-then-RL pipeline, the safest pattern is to apply the same mask during data collection, BC training, RL and deployment, and optionally add an auxiliary "is this action valid" head.

### Cited Findings
- **[Background]** Huang & Ontañón show that the usual way of masking invalid actions (setting their logits to minus infinity before the softmax) is still a valid policy gradient. They also show empirically that masking matters more "as the space of invalid actions grows". They evaluate several masking setups, including removing the mask after an agent has been trained with it. The code is public. Venue: FLAIRS 2022 (arXiv 2020). — [Huang & Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", arXiv:2006.14171 / FLAIRS-35, 2022](https://arxiv.org/abs/2006.14171); [FLAIRS proceedings](https://journals.flvc.org/FLAIRS/article/view/130584)
- **2026 (preprint):** Zabounidis, Siegelmann, Qadri, Kim, Stepputtis & Sycara name a failure mode of *unmasked* training, "valid action suppression". Gradients at visited states travel through shared parameters and push down valid actions at states not yet visited. Theorem 1, under linear features, gives an exponentially decaying bound on the probability of a valid action at a newly visited state. Entropy regularization keeps a floor but trades protection against sample efficiency, a tradeoff that masking removes. — [Zabounidis et al., "Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms", arXiv:2603.09090, Mar 2026](https://arxiv.org/html/2603.09090v1)
- Concrete numbers from the same paper:
  - The probability of the rarely valid `descend` action in Craftax fell from about 0.023 to below 0.0001 within 50M frames without masking.
  - Craftax-Hybrid with ground-truth masks: unmasked PPO 43.9, masked 45.6.
  - Masked policy plus an auxiliary feasibility-classification head with KL-balanced loss: 48.8.
  - Results use 4 seeds, and some differences fall within the reported spread. — [arXiv:2603.09090](https://arxiv.org/html/2603.09090v1)
- **Mask-dependence pitfall (same paper):** agents trained only with an oracle mask collapse to about -0.9 return when deployed with a *learned or predicted* mask. Agents that also had the feasibility head kept 28.6 (MLP), 36.4 (RNN) and 43.2 (Hybrid). Masking also leaves encoder features of valid and invalid states highly correlated (about 0.8). The feasibility head lowers this to about 0.4, so the network learns *why* an action is invalid. The paper does not test imitation or pretrained policies. It lists a BC head trained under predicted masks as future work. — [arXiv:2603.09090](https://arxiv.org/html/2603.09090v1)
- **2024+ masking work cited there:**
  - Stolz et al., continuous action masking (NeurIPS 2024)
  - Wang et al., learned state-specific masks via bisimulation (Algorithms 2024)
  - Theile et al., action mapping for constrained continuous RL (arXiv 2024)
  - Keane et al., "strategy masking" guardrails (arXiv 2025)
  - Liu et al., data-free physics-based action masks for assembly (below)

  — [arXiv:2603.09090 related work](https://arxiv.org/html/2603.09090v1)
- **2024–2025, most relevant to this project:** Liu, Chen, Zhao & Liu (CMU Intelligent Control Lab) train a deep RL policy to place unit primitives (Lego bricks) one at a time. At each step a *data-free* stability/feasibility mask, which needs no physics engine, removes physically infeasible placements. The authors say this "steers policy learning and ensures violation-free deployment". It reached 100% success on more than 250 Lego structures, while competing methods left more than 40 unsolved. Plans were executed by humans and robots. Code: github.com/intelligent-control-lab/PhysicsAwareCombinatorialASP. — [Liu et al., "Physics-Aware Combinatorial Assembly Sequence Planning using Data-free Action Masking", arXiv:2408.10162, v1 Aug 2024, revised to Mar 2025](https://arxiv.org/abs/2408.10162)
- **Stability-as-mask in construction RL, 2026:** the EPFL "Learning to Build" agent uses a state-dependent action space. Candidate placements that collide, leave the workspace, or fail a Rigid-Block Equilibrium (RBE) stability check are removed before Q-learning chooses. — [Wang, Kirschner, Rolland, Salamanca & Parascho, "Learning to Build: Autonomous Robotic Assembly of Stable Structures Without Predefined Plans", arXiv:2602.23934, 2026](https://arxiv.org/html/2602.23934)
- **[Background, from prior knowledge, URL not fetched]** Shielding as originally proposed: a shield synthesised from a temporal-logic safety spec plus a model of the environment overrides or restricts the learner's actions, either before the agent chooses ("preemptive", which is equivalent to masking) or after ("post-posed", which replaces unsafe actions). It guarantees safety during learning and deployment. — [Alshiekh, Bloem, Ehlers, Könighofer, Niekum & Topcu, "Safe Reinforcement Learning via Shielding", AAAI 2018, arXiv:1708.08611](https://arxiv.org/abs/1708.08611)
- **[Background]** Probabilistic shields (Jansen et al., CONCUR 2020) block actions whose probability of causing a violation exceeds the best achievable risk by more than a tolerance. Carr, Jansen, Junges & Topcu (AAAI 2023) extend shielding to partial observability. — [Jansen et al., "Safe Reinforcement Learning Using Probabilistic Shields", CONCUR 2020](http://www.cs.ru.nl/personal/nilsjansen/files/publications/jansen-et-al-concur-2020.pdf); [arXiv:1807.06096](https://arxiv.org/pdf/1807.06096)
- **2025:** Hamel-De le Court, Belardinelli & Goodall argue that training under a probabilistic shield keeps the agent safe while its cumulative reward stays close to optimal. This is listed as AAAI 2025 in a reference list, and I did not confirm that it matches the arXiv version. — [arXiv:2503.07671](https://arxiv.org/abs/2503.07671); see also [ProSh: Probabilistic Shielding for Model-free RL, arXiv:2510.15720](https://arxiv.org/pdf/2510.15720)
- **2025, CBFs:** Guerrier, Soma, Fouad & Beltrame model unsafe actions as continuous negative rewards *without* ending the episode. They report that standard RL "struggles" with this, because accumulated negative value in unsafe zones blocks learning. Combining RL with Control Barrier Functions (CBFs) "enhanc[es] learning outcomes" in simulation and on a real 4-wheel robot. The abstract gives no numbers. — [Guerrier et al., "Guided by Guardrails: Control Barrier Functions as Safety Instructors for Robotic Learning", arXiv:2505.18858, 2025](https://arxiv.org/abs/2505.18858)
- **Survey pointers:**
  - Guerrier, Fouad & Beltrame, survey on learning CBFs and their use in RL (2024)
  - Odriozola-Olalde et al., shielding review (2023)
  - Gu et al., broad safe-RL review (2022/2024)

  I did not open these surveys. — [reference list in arXiv:2503.07671](https://arxiv.org/html/2503.07671v1)
- **Redundant actions:** Zhong et al. show that redundant actions (not only invalid ones) cause repeated, useless attempts and can lead to task failure. They propose learned filtering that needs no rules or demonstrations. — [Zhong et al., "No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning", AAAI 2024, arXiv:2312.06258](https://ojs.aaai.org/index.php/AAAI/article/view/29652/31109)

### Inferences
- For this project, the holder-tree mask is a "preemptive shield" in Alshiekh et al.'s terms. Recent work supports keeping it on during RL. Removing it, or replacing it with a fall penalty, is the worse option: per Zabounidis et al., penalties lose ground to masking as the number of invalid actions grows.
- The mask-dependence result (masked-only agents collapse when the mask changes) matters directly here. If the mask is later loosened, for example to allow moves that are legal under a real ~3 N strength model but forbidden by the stricter holder tree, the policy will see a different action set than in training. One way to make the policy robust to mask changes is an auxiliary head that predicts mask or validity.
- For IL then RL: if the BC teacher's demonstrations always obeyed the mask, the BC policy should be trained on masked logits too. Otherwise BC probability mass sits on invalid actions and renormalisation distorts the policy. I found no paper testing this exact combination. It is my reasoning, not a cited result.
- Strength-aware masking is cheap and has precedent: Liu et al. mask Lego placements with an analytic stability check and no physics engine. A local cantilever/connector-force check could be added to the holder-tree mask the same way.

### Gaps
- I could not extract the numerical result of Huang & Ontañón's "masking removed" experiment (the PDF was not machine-readable here). My recollection is that performance drops when the mask is removed, but this is unverified.
- I found no paper that combines a hard safety mask with a BC-pretrained policy and then PPO fine-tuning, and that reports the interaction explicitly.
- The Gu et al. and Guerrier et al. surveys were not opened.

---

## 2. RL and learning for structurally constrained construction and assembly (stacking, bridges, arches, scaffolds)

### Takeaway
Learned construction with stability constraints is an active but small field. It is mostly EPFL (Parascho group), CMU and DeepMind, and mostly 2D or small 3D block worlds. The most relevant 2025 result for this project is Wang et al. (ACM TOG 2025). There, cooperating robots *act as the temporary supports*: some robots hold parts in place instead of scaffolding while others place. An RL policy with a GPU stability simulator, a curriculum for sparse reward, a GNN encoder and a disassembly-reversal trick plans online for a 62-part vault. Agents that *learn on their own* to place sacrificial or temporary supports that are later removed are almost undocumented. Counterweights and arch-like solutions do emerge in learned block building.

### Cited Findings
- **2025, key:** Robots replace dense scaffolding by holding parts in place during intermediate stages. An RL model generates *alternative assembly plans online* when parts arrive late or setups change, instead of recomputing offline. The parts:
  - a GPU-based stability simulator
  - curriculum training for sparse rewards
  - a GNN encoding assembly geometry
  - plans generated as *disassembly* sequences and then reversed into feasible assembly plans

  Demonstrated on a 62-part vault with 2–3 robots keeping it stable without scaffolding. — [Ziqi Wang, Wenjun Liu, Jingwen Wang, Gabriel Vallat, Fan Shi, Stefana Parascho & Maryam Kamgarpour, "Learning to Assemble with Alternative Plans", ACM Transactions on Graphics 44(4), 2025, DOI 10.1145/3730824](https://dl.acm.org/doi/10.1145/3730824); [EPFL Infoscience record](https://infoscience.epfl.ch/handle/20.500.14299/252688); [author PDF](https://kiki007.github.io/assets/pdf/Wang2025Learn2Assemble.pdf) (the PDF was too large to fetch, so I could not verify the experimental numbers.)
- **2026 (preprint):** "Learning to Build". Goal-conditioned deep Q-learning with successor features: a U-Net on 64x64 images, Q = successor features · reward field. Unit squares and trapezoids in a 10x10 2D world must reach target points around obstacles. Every candidate is checked for stability with RBE. Results:
  - One policy solves all 15 tasks (seed average about 13–14 of 15).
  - It uses 77% fewer search nodes than A* (5,763 vs 24,866), and A* failed one task.
  - Generalisation is about 80% under shifted targets and about 50% at interpolated layouts.
  - Real ABB arm: 12 of 14 attempted tasks succeeded (10 on the first try).
  - Emergent behaviour: a **counterweight block** to achieve an overhang (Task 6), and arch-like solutions.
  - The agent does **not** place temporary supports.
  - Failures: marginally stable structures collapsed in reality because binary RBE cannot capture marginal stability; reachability and gripper collisions were not modelled; errors accumulated over long horizons; construction noise was not modelled.
  - RBE evaluation is the training bottleneck, which makes 3D costly.

  — [Wang, Kirschner, Rolland, Salamanca & Parascho, arXiv:2602.23934, 2026](https://arxiv.org/html/2602.23934)
- **[Background, per related work in arXiv:2602.23934]**
  - Bapst et al. (ICML 2019) used graph-based relational agents to connect the ground to targets around obstacles. The blocks were *adhesive*, so the solutions do not transfer to dry stacking and were often "umbrella-like".
  - Funk et al. (CoRL 2021): hierarchical graph policy plus MCTS for stable 3D cube structures from fixed designs. The prompt's "2022" is the PMLR volume year.
  - Ghasemipour et al. (2022): large-scale structured RL for magnetic-block assembly with blueprints.
  - Kulshrestha & Qureshi (CoRL 2023): graph attention for multi-level rearrangement of up to 20 blocks.
  - **Vallat et al. (SCF 2023): RL for scaffold-free spanning structures, including arches over variable spans.**
  - Li et al. (ICRA 2022) and Wibranek et al. (eCAADe 2021): blueprint-free bridge assembly.

  — [arXiv:2602.23934 related work](https://arxiv.org/html/2602.23934); [Bapst et al., "Structured agents for physical construction", arXiv:1904.03177](https://arxiv.org/abs/1904.03177) [from prior knowledge, URL not fetched]; [Ghasemipour et al., arXiv:2203.13733](https://arxiv.org/abs/2203.13733)
- **[Background]** Bridge design and construction without a blueprint: deep RL plus curriculum learns a feasible bridge architecture for arbitrary cliff widths, then the robot builds it. — [Li et al., "Learning to Design and Construct Bridge without Blueprint", ResearchGate record, ~2021/ICRA 2022](https://www.researchgate.net/publication/353730159_Learning_to_Design_and_Construct_Bridge_without_Blueprint)
- **2024:** AssemblyComplete applies deep RL to 3D combinatorial (Lego) construction. It includes stability analysis of construction sequences and a real Lego bench assembly. — [AssemblyComplete, arXiv:2410.15469, Oct 2024](https://arxiv.org/pdf/2410.15469)
- **2024:** Data-free physics-aware masking reaches 100% on more than 250 Lego structures (see Section 1). — [arXiv:2408.10162](https://arxiv.org/abs/2408.10162)
- **2024 review:** a Springer review of generalisability in deep-RL assembly describes a bridge-building case with concrete bricks where PPO beat rule-based algorithms by 36% under changing site conditions. I saw this only in a search snippet and did not verify it in the text. — [J. Intelligent Manufacturing review, 2024, doi 10.1007/s10845-024-02546-5](https://link.springer.com/article/10.1007/s10845-024-02546-5)
- **2026:** "Rollback-free stable brick structure generation" is another recent stability-constrained generation paper. I saw only its title. — [arXiv:2605.06947](https://arxiv.org/html/2605.06947v1)

### Inferences
- The "robots as temporary supports" idea from Wang et al. 2025 maps well onto this project. In a self-reconfiguring cube swarm, the "temporary scaffold" *is* other bots, so learning to place supports amounts to learning which bots to park as pillars and when to release them. That is the same structure as Wang et al.'s holding robots.
- Two tricks from that line of work look directly reusable here: planning by disassembly from the goal (reversed for assembly) to get feasible sequences, and a curriculum for sparse rewards.
- The Learning to Build failure analysis warns that a *binary* stability check gives the policy no margin. Training under the real ~3 N limit would be more robust with a graded margin, for example connector load as a fraction of capacity, as a shaped penalty on top of the hard mask.
- No paper found shows an agent *discovering* temporary, later-removed supports purely from reward. The project's hope is plausible but unproven in the literature. Counterweights are the closest emergent behaviour.

### Gaps
- Exact quantitative results of Wang et al. 2025 (success rates, training time, baselines) were not retrievable because the PDF was too large and the ACM page returned 403.
- Vallat et al. 2023 (SCF) was not fetched.
- I found no 2024+ benchmark specifically for arches or Jenga-like stability with temporary scaffolding.

---

## 3. Collective construction: TERMES, ARMADAS / BILL-E, relative robots, 2024+ learned approaches

### Takeaway
The flagship collective-construction hardware systems (TERMES 2014, NASA ARMADAS 2024) use hand-designed rules and classical planners, not RL. They get robustness from **lattice periodicity** (the structure is the robot's ruler) and from reversible fastening. 2024–2026 RL work in collective construction is sparse and comes mainly from architecture or construction-robotics venues. A 2026 paper notes that most collective robotic construction (CRC) workflows still rely on handcrafted heuristics.

### Cited Findings
- **[Background, from prior knowledge, URL not fetched]** TERMES: termite-inspired climbing robots build user-specified 3D structures from bricks using only local rules compiled offline from the target shape. — [Werfel, Petersen & Nagpal, "Designing Collective Behavior in a Termite-Inspired Robot Construction Team", Science 343:754, 2014, doi 10.1126/science.1245842](https://doi.org/10.1126/science.1245842)
- **2024, ARMADAS (Science Robotics, Jan 2024):** fibre-reinforced composite truss voxels form an ultralight lattice metamaterial. Two types of mobile robots move over the outside and through the inside, transporting, placing and *reversibly fastening* voxels. They use "the intrinsic lattice periodicity for indexing and metrology". Reported:
  - a 256-cell assembly, plus disassembly and reconfiguration
  - density 0.0103 g/cc, strength 11.38 kPa, stiffness 1.1129 MPa. An earlier NASA record gives strength as 0.01117 MPa (about 11.17 kPa), so the numbers disagree slightly.

  — [Gregg, Catanoso, Formoso, Kostitsyna, Ochalek, Olatunde, Park, Sebastianelli, Taylor, Trinh & Cheung, "Ultralight, strong, and self-reprogrammable mechanical metamaterials", Science Robotics 9, Jan 2024 (NTRS record)](https://ntrs.nasa.gov/citations/20230005194)
- ARMADAS ground demo from press coverage, which is secondary: about 100 hours (reported elsewhere as 4.2 days of continuous runtime) to build a shelter of 256 voxels (30 cm each) with three robots. A SOLL-E robot places a voxel, then an MMIC-I unit fastens it. NASA supplied the shelter plans but not build instructions. Chief engineer Christine Gregg says the system corrects its own errors without machine vision. — [Freethink](https://www.freethink.com/space/space-robots-armadas); [Interesting Engineering](https://interestingengineering.com/innovation/robots-resilient-structures-adaptable-metamaterials); [NASA patent TOP2-321](https://technology.nasa.gov/patent/TOP2-321); SOLL-E paper: [IROS 2023 (NTRS)](https://ntrs.nasa.gov/api/citations/20230011353/downloads/IEEE_IROS_2023_SOLLE_Final.pdf)
- **2024 companion paper (NASA):** Gregg & Cheung simulate how voxel geometry and assembly resolution affect precision. Increasing plate thickness "quickly collapses RMS error towards a limit that is on the order of the error of the constituent parts". — [NTRS 20240000666](https://ntrs.nasa.gov/citations/20240000666)
- **ICRA 2024, MIT CBA:** self-reconfigurable robots for collaborative discrete lattice assembly. The assembler robot is itself built from modules compatible with the voxel material it assembles ("relative robots" line). I saw no learning component in the excerpt. — [MIT CBA, "Self-Reconfigurable Robots for Collaborative Discrete Lattice Assembly", ICRA 2024](https://cba.mit.edu/docs/papers/24.05.ICRA.pdf)
- **2024 (architecture):** Hosmer et al. combine modular robotic materials with deep multi-agent RL for "collaborative modular robotic assembly of reconfigurable building parts", in three case studies. — [Hosmer, Mutis, Gheorghiu, Siedler, He & Erdincer, "Autonomous ecologies of construction", Int. J. Architectural Computing, 2024, doi 10.1177/14780771241287827](https://journals.sagepub.com/doi/10.1177/14780771241287827)
- **2025:** MARL with inverse kinematics for collaborative construction robots. The authors report efficient collaboration and collision avoidance. — [Duan & Zou, "Enhancing construction robot collaboration via multiagent reinforcement learning", J. Intelligent Construction, 2025](https://www.sciopen.com/article/10.26599/JIC.2025.9180089)
- **2024 (planning, not RL):** the ReRamp algorithm builds taller and larger ramped block structures (TERMES-style) than earlier planners, including a single-storey house plan. — Rameš & Surynek, "Reaching New Heights", Oct 2024 chapter (search snippet only; no stable URL captured)
- **Benchmark:** CraftEnv (AAMAS 2023) is a collective robotic construction environment for MARL. It is cited by a 2026 paper on real-world craft robots. — [arXiv:2602.21119](https://arxiv.org/pdf/2602.21119)
- **2026 (LLM-based, included only for its opinion):** the authors state that most CRC workflows still depend on handcrafted heuristics and that RL "is costly to train and sensitive to the environment". — [Springer Construction Robotics, 2026, doi 10.1007/s41693-026-00216-x](https://link.springer.com/article/10.1007/s41693-026-00216-x)

### Inferences
- ARMADAS shows that a lattice with reversible fasteners and lattice-indexed metrology can make hardware assembly reliable without learning. Learning has not yet been the bottleneck in real collective construction; mechanics, fastening and error correction have.
- Strength concerns in ARMADAS are handled by strong bolted connections, unlike the project's weak ~3 N connectors. This project's regime, where structural limits shape the plan, is therefore less explored than ARMADAS's.

### Gaps
- I did not access the ARMADAS Science Robotics full text. The 2025 NTRS slide deck PDF could not be parsed. I could not confirm how ARMADAS plans assembly order, or whether it checks intermediate-structure stability.
- No 2024+ paper on BILL-E or Jenkins' relative robots was surfaced.
- I found no RL result on TERMES-like hardware.

---

## 4. Modular self-reconfigurable robots with learned control: what has worked?

### Takeaway
Lattice reconfiguration theory (sliding/pivoting cubes, SoCG) and practical reconfiguration planners in 2024–2025 are almost entirely classical: search, PSO plus RRT, and provable algorithms. Learned control has worked in 2D pivoting-cube ensembles with local sensing (2025/2026) and in freeform or locomotion modular robots. I found no 2024+ RL paper on 3D lattice reconfiguration that enforces connectivity as a hard constraint. This project's combination (3D lattice, connectivity mask, structural load) appears new.

### Cited Findings
- **2025/2026:** Dobreva, Blazquez, Grover, Izzo, Qin & Dold train per-cube neural controllers with RL for 2D homogeneous pivoting-cube ensembles. Cube *selection* (which cube moves) is still globally coordinated. Findings:
  - "Local sensing is sufficient for effective global reconfiguration".
  - More global information makes reconfiguration faster.
  - Near-optimal reconfiguration is reached with only nearest-neighbour interactions if information is passed several times.
  - Building grid symmetries into the network (geometric deep learning) gives only minor training benefits but allows smaller models.
  - Transfer to sliding cubes is proposed but not shown.

  — [Dobreva et al., "Decentralised self-organisation of pivoting cube ensembles", arXiv:2509.03140, Astrodynamics (Springer) 2026](https://www.arxiv.org/abs/2509.03140)
- **[Background] 2023:** multi-agent RL (an "altruism" PPO variant) for distributed self-reconfiguration of *freeform* modules. The authors argue that lattice-based algorithms do not suit continuous configuration spaces. — [IJCAI 2023, "Learning to Self-Reconfigure for Freeform Modular Robots via Altruism Proximal Policy Optimization"](https://ijcai.org/proceedings/2023/0610.pdf)
- **2024 theory (SoCG):** in the sliding-cubes model any two connected shapes can be reconfigured into each other in O(n²) moves *in place*, keeping connectivity. A 2025 follow-up gives input-sensitive bounds with O(1) amortised computation per move. — [cited in "Input-Sensitive Reconfiguration of Sliding Cubes", arXiv:2507.04170, 2025](https://arxiv.org/html/2507.04170); background: [Abel & Kominers, arXiv:0802.3414](https://arxiv.org/html/0802.3414)
- **2024–2025 theory:** "Sliding squares in parallel" (arXiv:2412.05523) and "Distributed Rhombus Formation of Sliding Squares" (arXiv:2508.09638). I saw titles only. — [arXiv:2508.09638](https://arxiv.org/pdf/2508.09638)
- **2024:** fault-tolerant self-reconfiguration of modular robots with locked joints uses PSO to choose key configurations and RRT for paths, not RL. — [Qi et al., Advanced Intelligent Systems, 2024, doi 10.1002/aisy.202300774](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300774)
- **2025:** environment exploitation for self-reconfiguration, a planning-based study. — [arXiv:2508.01829](https://arxiv.org/pdf/2508.01829)
- **2025 (locomotion, not reconfiguration):** decentralised per-module RL (SAC for wheel modules, PPO for 7-DoF limbs) for modular lunar robots, with zero-shot generalisation to unseen configurations. — [arXiv:2510.20347](https://arxiv.org/html/2510.20347v1)
- **2025:** modular tensegrity blocks that self-assemble. The authors *suggest* that RL and MPC could make gait and connection strategies adaptive in the future. This is a direction, not a result. — [Nature Communications 2025, doi 10.1038/s41467-025-60982-0](https://www.nature.com/articles/s41467-025-60982-0)
- **[Background, from prior knowledge, URL not fetched]** M-Blocks: self-reconfiguring cubes that pivot using an internal flywheel and magnetic edges (Romanishin, Gilpin & Rus, IROS 2013; M-Blocks 2.0, IROS 2019). Control was hand-designed. Magnetic, bond-limited connections are known to restrict which pivots succeed. — [Romanishin et al., "M-blocks: Momentum-driven, magnetic modular robots", IROS 2013](https://ieeexplore.ieee.org/document/6696971)

### Inferences
- What has worked with learning: 2D, local observations, shared per-module policy, and global (or trivial) selection of the mover. That matches this project's "local policy plus map compass" setup. The Dobreva et al. result that symmetry-equivariant networks gave only minor gains suggests not prioritising equivariance.
- What has not been shown: 3D, hard connectivity under gravity, structural load limits, and fully decentralised mover selection with learned control.
- The O(n²) universal in-place sliding-cube algorithm gives a classical fallback or "un-jam" teacher. It does not consider gravity or connector strength.

### Gaps
- I found no paper by "Dutta", "Tucci" or "Zhu" on RL for lattice reconfiguration in 2022–2025. Those names from the prompt could not be matched to specific papers in my searches.
- No catom or Claytronics RL paper (2024+) surfaced.

---

## 5. Fine-tuning imitation-pretrained policies with RL: consensus, pitfalls, reward design, multi-agent credit assignment

### Takeaway
The 2024–2025 consensus is that RL fine-tuning of a BC or pretrained policy is effective, but it reliably causes **forgetting of pretrained skills in states the fine-tuning phase does not visit early**. Standard retention tools fix most of it: a BC auxiliary loss on teacher data, a KL penalty to the teacher, or kickstarting. A second route is residual RL, which freezes the BC policy and learns small corrections. For many agents sharing one policy, on-policy PPO with full parameter sharing and a centralised value function (MAPPO) is a strong, well-tested default. Sharing parameters helped a lot in ablations.

### Cited Findings
- **ICML 2024 (Spotlight):** fine-tuning RL models erodes pre-trained capabilities in states of the downstream task that early fine-tuning does not visit. The effect is often "catastrophic". "Standard knowledge retention techniques mitigate the problem". On NetHack (Human Monk), retention-aided fine-tuning exceeded 10K points versus the previous best of about 5K for neural models. Search snippets of the appendix indicate that a BC-loss retention method was robust across network sizes, while EWC failed for 2-layer networks. — [Wołczyk, Cupiał, Ostaszewski, Bortkiewicz, Zając, Pascanu, Kuciński & Miłoś, "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem", ICML 2024, arXiv:2402.02868](https://arxiv.org/abs/2402.02868)
- **2025 robotics (excavation):** removing the KL penalty from RL fine-tuning of a pretrained policy caused "substantial degradation of the move arm skill" across all three adaptation scenarios. With KL, drift from the pretrained prior was limited and out-of-distribution performance was preserved. A plain MLP-PPO baseline still forgot the original configurations after about 115M interactions. — [ExT, arXiv:2509.14992, 2025](https://arxiv.org/pdf/2509.14992)
- **2025 humanoids:** unconstrained fine-tuning eroded the motion style learned in pre-training. The fix was a regularised loss that keeps distilling from the expert controller. — [PPF, arXiv:2504.09833, 2025](https://arxiv.org/pdf/2504.09833)
- **2025:** "RL's Razor": forgetting tracks the KL divergence between the fine-tuned and base policy on the new task. On-policy RL tends to find low-KL solutions. — [arXiv:2509.04259 (via Emergent Mind summary)](https://www.emergentmind.com/papers/2509.04259)
- **[Background, as cited 2026]** Video PreTraining (Baker et al., 2022): a KL constraint to the pretrained model was key. Without it, RL fine-tuning collapsed after early progress. — [cited in arXiv:2607.19399](https://arxiv.org/pdf/2607.19399)
- **Residual RL, Dec 2024:** Policy Decorator learns a model-agnostic residual policy on top of large IL models (Behavior Transformer, Diffusion Policy) with "controlled exploration", across 8 tasks and 2 benchmarks. — [arXiv:2412.13630](https://arxiv.org/abs/2412.13630). A 2026 comparison found that it recovers BC performance on easy tasks but needs much more interaction on harder ones. — [Q2RL, arXiv:2605.05172](https://arxiv.org/pdf/2605.05172)
- **Residual RL, 2025:** ResFiT freezes an action-chunked BC policy and learns per-step residual corrections with off-policy RL from sparse rewards. On real robots, success rose from about 14–23% to about 64% within about 15–76 minutes of interaction. — [ResFiT, "Residual Off-Policy RL for Finetuning Behavior Cloning Policies", arXiv:2509.19301](https://arxiv.org/html/2509.19301v2)
- **IBRL:** uses the IL policy's actions as alternative proposals for exploration and for critic targets, instead of residual RL. — [described in arXiv:2509.19301](https://arxiv.org/pdf/2509.19301)
- **Sparse reward in construction:** Wang et al. (TOG 2025) needed a *curriculum* to train under sparse assembly rewards. — [DOI 10.1145/3730824](https://dl.acm.org/doi/10.1145/3730824)
- **Shaped reward in construction:** Learning to Build used a dense, Gaussian-smoothed target reward field plus a small per-block penalty, which cut average assembly size from about 10 to 5–6 blocks. — [arXiv:2602.23934](https://arxiv.org/html/2602.23934)
- **Multi-agent, shared policy:** MAPPO (PPO with a centralised value function, with both policy and value networks shared) is competitive with or better than off-policy MARL on MPE, SMAC, Google Research Football and Hanabi, with little tuning. In ablations on hard SMAC maps, shared parameters reached median win rates of about 90% (MMM2) and 88% (6h_vs_8z), versus about 13% and 11% without sharing. — [Yu, Velu, Vinitsky, Gao, Wang, Bayen & Wu, "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", NeurIPS 2022 Datasets & Benchmarks, arXiv:2103.01955](https://arxiv.org/pdf/2103.01955)
- **Credit assignment:** PRD-MAPPO decouples agents from teammates that do not affect their expected reward, and CoPPO claims dynamic credit assignment. Both are from secondary descriptions and unverified. — [listing for arXiv:2103.01955 related work](https://arxiv.deeppaper.ai/papers/2103.01955v4/similar)

### Inferences
- For this project, the forgetting result means that fine-tuning on hard bridge or arch shapes could degrade shapes the BC policy already builds. A BC loss on the existing DAgger dataset, or a KL penalty to the frozen BC policy, is the low-risk default. A regression test on already-solved shapes should be kept during RL.
- Residual RL (frozen BC plus a learned correction) matches discrete lattice moves less well than continuous control. A discrete analogue is to add learned logit offsets to the frozen BC logits under the same mask. That is my suggestion; no paper here tests it.
- Given a construction curriculum (Wang et al.) and the success of dense target fields (Learning to Build), a reasonable design is a dense "voxels filled correctly" term plus a sparse completion bonus, with a curriculum over span length for bridges.
- MAPPO-style full sharing plus a centralised critic fits a homogeneous swarm. Credit assignment may still be hard, because only a few bots act as temporary supports that are later released.

### Gaps
- I found no paper that does IL then RL specifically for lattice modular robots or construction swarms.
- Evidence on how the KL coefficient interacts with a hard action mask was not found.

---

## 6. Sim-to-real for modular/swarm robots with weak connectors: what is typically modelled or ignored?

### Takeaway
Simulators for learned construction usually reduce physics to a **binary static-equilibrium check** (rigid-block equilibrium) or a data-free feasibility rule. They usually ignore placement noise, accumulated tolerance error, connector compliance, marginal stability, dynamics during moves, and robot reachability. The 2026 EPFL real-robot study traced its failures to exactly these gaps. Hardware systems that work (ARMADAS) avoid the problem with strong reversible fasteners and lattice self-alignment, which bounds error near part tolerance. I found very little on modelling weak magnetic or latching connector strength for learned controllers.

### Cited Findings
- **Modelled:** binary RBE stability for each candidate placement. **Ignored:** construction noise, reachability and gripper collisions, marginal stability. Real-robot failures came from marginally stable structures collapsing (Tasks 9 and 11 needed retries), reachability (Task 4) and accumulated placement error (Task 7). The authors propose a coupled rigid-block solver (Kao et al. 2022) as future work. — [arXiv:2602.23934](https://arxiv.org/html/2602.23934)
- **Modelled:** stability through a GPU-based stability simulator, used inside RL training for robot-supported vault assembly. Its friction or contact model was not verified (PDF not retrievable). — [Wang et al., TOG 2025](https://dl.acm.org/doi/10.1145/3730824)
- **Modelled:** a data-free physics feasibility rule for Lego, needing no physics engine. Plans executed on real robots. — [arXiv:2408.10162](https://arxiv.org/abs/2408.10162)
- **Hardware approach:** ARMADAS uses lattice periodicity for indexing and metrology, plus reversible fastening. A companion simulation shows assembly RMS error collapses towards the error of the constituent parts as plate thickness grows. Precision is part-limited, not accumulated. — [Science Robotics 2024 (NTRS)](https://ntrs.nasa.gov/citations/20230005194); [NTRS 20240000666](https://ntrs.nasa.gov/citations/20240000666)
- **2025/2026:** the pivoting-cube RL work is purely 2D and kinematic. Transfer to hardware (sliding cubes, CubeSat swarms) is proposed but not demonstrated. — [arXiv:2509.03140](https://www.arxiv.org/abs/2509.03140)

### Inferences
- For a ~3 N connector, the common binary check would hide marginal cases, as with RBE. A sim that computes per-connector load (moment arm × weight of the cantilevered sub-tree) and trains with a safety margin, or randomises connector strength (domain randomisation), would target the failure mode EPFL saw.
- Dynamic loads during slide or pivot moves (momentum and impulse during M-Blocks-style pivots) are typically ignored in learned-reconfiguration sims. For weak connectors this is likely the largest unmodelled effect. This is an inference; I found no source that quantifies it.

### Gaps
- I found no 2024+ paper that models connector strength (magnetic or latch force limits) inside an RL training loop for lattice modular robots.
- I found no quantitative sim-to-real study for M-Blocks-style pivoting under load.
