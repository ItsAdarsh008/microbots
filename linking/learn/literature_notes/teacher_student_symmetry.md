# Privileged teacher to local student distillation, DAgger, the imitation gap, CTDE, and symmetry (augmentation vs equivariance)

Scope note: about 20 search and fetch calls. Most 2024+ items were checked against arXiv abstract pages. Items marked "background, not re-fetched" are pre-2024 foundational papers cited from prior knowledge, with their canonical arXiv URLs. The report writer should treat their exact venue and details as needing a quick check. Several 2025-2026 items are arXiv preprints without a confirmed peer-reviewed venue, and this is flagged on each one.

Project framing used throughout:
- **Teacher:** a hand-written rule teacher that can see the global crawl map (a whole-grid distance field).
- **Student:** one MLP shared by every bot. Each bot sees a local 5x5x5 window, the target map, a relayed hop count, and a compass.
- **Training:** behaviour cloning (BC), then 2 rounds of DAgger.
- **Results:** about 65% agreement with the stochastic teacher on held-out decisions, but every test shape gets built. With a weaker local signal, the student failed (arch 2/5).

## Q1. Privileged teacher / "learning by cheating" / teacher-student distillation: when does it work, when does it fail?

### Takeaway
Two-stage "train a teacher on privileged state, then distill it into a student" is the default recipe in legged locomotion and is widely used in driving, drones and manipulation. Since 2024 the consensus is that it works well when the privileged information is inferable from the student's observation history. It degrades, sometimes badly, when it is not: the student receives conflicting labels for the same observation, then drifts off-distribution. The 2024-2026 response has been to make the teacher student-aware (penalized for actions the student cannot reproduce), to co-train teacher and student, or to add RL on the student's own reward.

### Cited Findings
**Foundational work (background, not re-fetched)**
- Chen, Zhou, Koltun and Krähenbühl, "Learning by Cheating," CoRL 2019. A privileged agent sees ground-truth layout and traffic, then teaches a vision-based sensorimotor student. This decomposition made the student problem much easier and gave large gains in the CARLA driving benchmark. — [arXiv 1912.12294](https://arxiv.org/abs/1912.12294)
- Lee, Hwangbo, Wellhausen, Koltun and Hutter, "Learning quadrupedal locomotion over challenging terrain," Science Robotics 2020. The teacher has privileged terrain and contact information; a proprioceptive student is distilled with a temporal (TCN) history encoder. This is the canonical legged teacher-student recipe. — [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
- Kumar, Fu, Pathak and Malik, "RMA: Rapid Motor Adaptation for Legged Robots," RSS 2021. The teacher is conditioned on an encoded privileged environment extrinsics vector. The student's adaptation module regresses that latent from its state-action history. The design premise is that the privileged information is recoverable from history. — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)
- Miki, Lee, Hwangbo, Wellhausen, Koltun and Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild," Science Robotics 2022. A privileged teacher is distilled into a student with a recurrent belief encoder that fuses noisy exteroception with proprioception. — [arXiv 2201.08117](https://arxiv.org/abs/2201.08117)

**Same recipe in 2024-2026 work**
- PA-LOCO (2024) describes the same two phases. A teacher is trained with privileged information that is not available at deployment, then a student replicates the teacher using onboard sensors. — [PA-LOCO, arXiv 2407.04224](https://arxiv.org/pdf/2407.04224)
- Privileged imitation learning is now a named, distinct setting: an expert policy with privileged information already exists, and a partially observed student imitates it. — [Hu et al., Scaffolder, ICLR 2024, project page](https://penn-pal-lab.github.io/scaffolder/)

**Failure mode: information asymmetry**
- Kim, Chin, Vasudev and Choudhury (2025) put information asymmetry at the centre. The student cannot directly access the teacher's state space, which leads to distributional shift and policy degradation, and the same student observation can receive conflicting labels, which destabilizes learning. — [Kim et al., "Distilling Realizable Students from Unrealizable Teachers," arXiv 2505.09546, 2025 (venue not stated on arXiv)](https://arxiv.org/abs/2505.09546); conflicting-labels wording from [ResearchGate listing](https://www.researchgate.net/publication/391741702_Distilling_Realizable_Students_from_Unrealizable_Teachers)
- Kim et al. sort existing fixes into two groups, and say both are inefficient:
  - modify the teacher so it produces "realizable but sub-optimal demonstrations";
  - have the student explore to recover the missing information.
- Their own remedy has two parts:
  - an IL method that adaptively decides when the student should query the teacher;
  - an RL method that chooses where to initialize training (reset from recovery states).
  They report significant gains over standard teacher-student baselines in simulation and on real robots; the abstract gives no numbers. — [arXiv 2505.09546](https://arxiv.org/abs/2505.09546)
- Messikommer, Xing, Aljalbout and Scaramuzza, "Student-Informed Teacher Training," ICLR 2025. Teacher-student asymmetry arises because the teacher is trained without checking whether the student can reproduce its behaviour. Their fix, derived from an imitation-learning performance bound, has two parts:
  - add an approximate teacher-student action difference as a penalty in the teacher's reward;
  - add a supervised teacher-student alignment step.
  They demonstrate this on maze navigation, vision-based quadrotor flight and manipulation. — [arXiv 2412.09149](https://arxiv.org/abs/2412.09149)
- Wang, Luo, Zhang and Chen, "CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion," IEEE RA-L 2024. Teacher and student are trained concurrently under a modified PPO that uses rollouts from both policies, instead of RL followed by supervised distillation. They report up to about 20% lower velocity-tracking error than two-stage teacher-student training on blind locomotion, for quadrupeds and point-foot bipeds. (Secondary summaries disagree on exactly which networks are shared.) — [arXiv 2405.10830](https://arxiv.org/abs/2405.10830); [IEEE Xplore](https://ieeexplore.ieee.org/document/10670293/)
- A 2026 humanoid paper (MuGen) reports that distilling privileged teachers "often encounters a capability drop due to the lack of privileged information." It mitigates this with a latent codebook shared between teacher and student. — [MuGen, arXiv 2605.24592](https://arxiv.org/pdf/2605.24592)

**Adjacent field: LLMs (2025-2026), same pattern**
- Privileged on-policy distillation can "pull students toward distributions they can't reach locally." AR-OPD anchors the student to a *partially* privileged teacher and treats oracle foresight as a controlled residual. — [AR-OPD, arXiv 2606.10385](https://arxiv.org/html/2606.10385v1)
- A survey reports that the student-teacher gap widens as the teacher's privileged context grows. — [Survey of On-Policy Distillation, arXiv 2604.00626](https://arxiv.org/html/2604.00626v3)
- SCoRe has the student generate trajectories while the teacher corrects only the earliest mistake. This is a DAgger-like "student drives, teacher corrects" scheme. — [arXiv 2509.14257](https://arxiv.org/abs/2509.14257)

### Inferences
- The consensus condition for success is that the privileged variable is a function of, or recoverable from, the student's observation history; RMA's and Miki's history and belief encoders are built on this. The microbots result fits this pattern:
  - The global crawl-distance field is not recoverable from a 5x5x5 window alone. That is the "weak local signal" failure (arch 2/5).
  - The relayed hop count is a cheap distributed approximation of that field, so adding it made the privileged information largely inferable again.
- The project plan (keep the global teacher, keep DAgger) matches the mainstream recipe. The 2024-2025 literature says the next lever, if failures recur, is to make the teacher *student-aware* rather than drop it. Two options:
  - penalize teacher actions the local student cannot reproduce (Messikommer et al. 2025);
  - restrict the teacher's guidance to states the student can recover from (Kim et al. 2025).
- With a hand-written rule teacher, "student-aware" could mean using a crawl map computed only from information that relayed hop counts can carry. This is an inference, not a published recipe for lattice swarms.

### Gaps
- No 2024+ survey was found that quantifies, across many tasks, when teacher-student distillation beats asymmetric actor-critic or end-to-end RL. The searches returned only individual-paper claims.
- No published teacher-student distillation work for modular or lattice self-reconfiguring robots turned up in these searches.
- No quantitative numbers were retrieved for Kim et al. 2025 or Messikommer et al. 2025; only abstract-level claims.

## Q2. The imitation gap / asymmetric information: what remedies exist?

### Takeaway
"Impossibly good experts" and the "imitation gap" name one phenomenon. Behaviour cloning on a privileged expert recovers only the expert's action *marginalized over the hidden information*, which can be badly suboptimal. Remedies fall into five families:
1. **Mix in RL reward:** ADVISOR, TGRL.
2. **Make the teacher imitable:** GPO, student-informed teacher training.
3. **Use privileged information only in the critic or world model:** asymmetric actor-critic, Scaffolder, Informed AAC.
4. **Give the student memory or sequence models:** Swamy et al.
5. **Gate when to query the teacher:** Kim et al. 2025.

Theory from 2024 gives a condition, the "deterministic filter" condition, under which plain expert distillation is provably efficient.

### Cited Findings
**Foundational work (background, not re-fetched)**
- Swamy, Choudhury, Bagnell and Wu, "Sequence Model Imitation Learning with Unobserved Contexts," NeurIPS 2022. With unobserved context, on-policy methods such as DAgger plus a history-conditioned (sequence) policy can recover the expert's performance where off-policy BC on histories can fail, through a "latching" effect. — [arXiv 2208.02225](https://arxiv.org/abs/2208.02225)
- Weihs et al., ADVISOR. "When the teaching agent makes decisions with access to privileged information that is unavailable to the student, this information is marginalized during imitation learning." ADVISOR dynamically weights IL and RL losses per state. This is NeurIPS 2021 work: one search snippet dated it 2024, which conflicts with the original venue. — [Weihs et al., "Bridging the Imitation Gap by Adaptive Insubordination," arXiv 2007.12173](https://arxiv.org/abs/2007.12173)
- Walsman, Zhang, Choudhury, Fox and Farhadi, "Impossibly Good Experts and How to Follow Them," ICLR 2023. Experts with extra information are framed as beyond what any learner can reproduce. The goal becomes the best policy a restricted learner can reach, with gains reported on MiniGrid. — [ICLR 2023 poster page](https://iclr.cc/virtual/2023/poster/11786)
- Shenfeld, Hong, Tamar and Agrawal, "TGRL: An Algorithm for Teacher Guided Reinforcement Learning," ICML 2023. It gives a principled way to balance automatically, per stage, when to follow the teacher and when to use reward. It is aimed at cases where the agent cannot mimic the teacher. — [arXiv 2307.03186](https://arxiv.org/abs/2307.03186)
- Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning," 2017. The critic gets full state and the actor gets observations. — [arXiv 1710.06542](https://arxiv.org/abs/1710.06542)
- Baisero and Amato, 2021-22. The original asymmetric critics are biased. They propose an unbiased asymmetric actor-critic that conditions the critic on history and state. — [arXiv 2105.11674](https://arxiv.org/pdf/2105.11674)

**2024-2026 work**
- Hu, Springer, Rybkin and Jayaraman, "Privileged Sensing Scaffolds Reinforcement Learning" (Scaffolder), ICLR 2024 spotlight.
  - **Method:** puts privileged sensing into the world model, exploration policy and critic of a DreamerV3 agent, rather than only into a teacher.
  - **Benchmark:** introduces the 10-task Sensory Scaffolding Suite (S3).
  - **Result:** the project page says it "frequently performs comparably even to policies that have test-time access to the privileged sensors."
  - A third-party page claims it bridges about 79% of the gap between impoverished and fully informed agents; this could not be verified against the paper.
  — [arXiv 2405.14853](https://arxiv.org/html/2405.14853v1); [project page](https://penn-pal-lab.github.io/scaffolder/)
- Li, Xie and Lu, "Guided Policy Optimization under Partial Observability," arXiv 2505.15418, May 2025, revised March 2026; no venue listed.
  - **Method:** co-trains a privileged "guider" and a learner, keeping the guider aligned with the learner, which is trained mainly by IL. This yields a "possibly good" teacher that stays inside the learner's imitable region.
  - **Claims:** optimality "comparable to direct RL," and significantly better results than existing methods on partially observable or noisy continuous control and memory tasks.
  — [arXiv 2505.15418](https://arxiv.org/abs/2505.15418)
- Kim, Chin, Vasudev and Choudhury, 2025. The remedy is query-when-necessary IL plus RL resets from recovery states, which keeps the student on paths recoverable in its own observation space (details in Q1). — [arXiv 2505.09546](https://arxiv.org/abs/2505.09546)
- Messikommer et al., ICLR 2025. A teacher reward penalty on teacher-student action mismatch, plus an alignment step (details in Q1). — [arXiv 2412.09149](https://arxiv.org/abs/2412.09149)
- "Provable partially observable RL with privileged information," arXiv 2412.00985 (late 2024). It identifies a "deterministic filter condition" under which expert distillation has polynomial sample and computational complexity, and gives a belief-weighted asymmetric actor-critic with polynomial sample and quasi-polynomial computational complexity. Only a secondary summary was seen, so verify against the paper. — [Emergent Mind summary of arXiv 2412.00985](https://api.emergentmind.com/papers/2412.00985)
- Ebi et al., "Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access," ICML 2026, PMLR 306.
  - The critic can condition on any state-dependent privileged signal (not only full state) and the policy gradient stays unbiased.
  - It offers a pre-training dependence test and a post-hoc value-prediction test for ranking signals.
  - Well-chosen partial signals "can match or outperform full-state asymmetric baselines while relying on strictly less state information."
  — [arXiv 2509.26000](https://arxiv.org/pdf/2509.26000); [PMLR](https://proceedings.mlr.press/v306/ebi26a.html)
- PIGDreamer (2025) formalizes "Asymmetric Constrained POMDPs" for safe RL with privileged world models. — [arXiv 2508.02159](https://arxiv.org/html/2508.02159v1)

### Inferences
- **The 65% agreement figure.** A stochastic teacher that breaks ties randomly among equally good moves puts an upper bound on exact-match agreement well below 100%. 65% agreement plus full task success suggests much of the "disagreement" is among teacher-equivalent actions, not a true imitation gap.
  - Better diagnostics would be:
    - whether the student's action is in the teacher's *set* of acceptable actions;
    - the cross-entropy or KL to the teacher's action distribution;
    - task success and connectivity violations.
  - This is an inference from how stochastic experts bound agreement, not a published result on this setup.
- **The arch 2/5 failure** under a weaker local signal is the textbook imitation gap: the teacher's action depended on the global field, which the student could not infer. Remedy families 1-5 above all apply. The cheapest for this project:
  - **(a)** Keep DAgger and add a small reward-based fine-tuning stage (an ADVISOR or TGRL-style IL/RL mix).
  - **(b)** Give the student a short memory, a recurrent state or a few steps of history (Swamy et al.).
  - **(c)** Constrain the rule teacher to use only information the relay can provide (GPO, "possibly good" teacher).
- **Informed AAC** suggests a fit for a later RL fine-tune: the critic could use the global crawl map as a privileged signal while the actor stays local.

### Gaps
- Exact venues could not be confirmed for Kim et al. 2025 and GPO (arXiv only), or for arXiv 2412.00985 (secondary summary only).
- No 2024+ head-to-head benchmark across these remedy families on a common suite was found. Scaffolder's S3 is the closest common benchmark.

## Q3. CTDE in multi-agent RL as a parallel: consensus on value and pitfalls

### Takeaway
CTDE (centralized training, decentralized execution) is still the dominant MARL paradigm in 2024-2025 surveys. It is valued for scalability and for needing no communication at execution, while allowing global state and other agents' information during training. Critiques say:
- it "can often perform well" but is not universally best;
- centralized critics can add variance or bias;
- independent per-agent policies may not be "centralized enough" to coordinate.

The microbots setup (a shared local policy trained from a global-information teacher) is a CTDE-style setup: the privileged information enters through the teacher's labels rather than through a centralized critic.

### Cited Findings
- Amato (2024), "An Introduction to Centralized Training for Decentralized Execution in Cooperative Multi-Agent Reinforcement Learning," arXiv 2409.03052.
  - CTDE is the most common of three paradigms (CTE, CTDE, DTE).
  - It is the only paradigm with a separate training phase, so training can use any available information, such as other agents' policies or the true state.
  - It is more scalable than CTE, needs no execution-time communication, and "can often perform well," which implies it is not always best.
  — [arXiv 2409.03052](https://arxiv.org/abs/2409.03052)
- A 2025 transportation MARL survey frames CTDE as balancing centralized learning (coordination, faster convergence, sample efficiency) against decentralized flexibility and scalability. — [arXiv 2508.20315](https://arxiv.org/pdf/2508.20315)
- A 2020-2025 MARL survey (Zenodo, not peer-reviewed) organizes the field around CTDE: value decomposition, actor-critic extensions, communication, mean-field, offline MARL. — [Zenodo 17138304](https://zenodo.org/records/17138304)
- Critique, CADP (2023): CTDE assumes independent agent policies, which keeps agents from sharing global cooperative information during training. The proposed fix is centralized advising during training, then pruning it for decentralized execution. — [arXiv 2305.17352](https://arxiv.org/abs/2305.17352)

**Background, not re-fetched**
- Lyu, Xiao, Daley and Amato, "Contrasting Centralized and Decentralized Critics in Multi-Agent RL," AAMAS 2021. Centralized critics are not strictly better: they can add variance and bias issues relative to decentralized critics. — [arXiv 2102.04402](https://arxiv.org/abs/2102.04402)
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO), NeurIPS 2022 Datasets & Benchmarks. Parameter-shared PPO with a centralized value function is a strong baseline. — [arXiv 2103.01955](https://arxiv.org/abs/2103.01955)
- McClellan et al., NeurIPS 2024 (see Q5). Equivariant parameter-shared GNN policies in MARL are one way to inject structure into CTDE. — [arXiv 2410.02581](https://arxiv.org/abs/2410.02581)

### Inferences
- MARL's known CTDE pitfall maps directly onto the imitation gap: a decentralized policy cannot represent behaviour that needs joint or global information. A centralized critic or teacher can ask for coordination the local actors cannot express.
- The microbots relayed hop count acts as learned or engineered communication. MARL research on communication within CTDE is the relevant adjacent literature; it was not covered in depth here.
- A shared MLP across all bots is the standard parameter-sharing CTDE choice (MAPPO-style). Its known limitation is homogeneous behaviour, unless agents receive distinguishing inputs such as the compass or hop count.

### Gaps
- No dedicated 2024-2026 *critique* survey of CTDE was found. The critiques located are individual papers: CADP 2023, and Lyu et al. 2021 as background.
- The CTDE literature found does not cover CTDE for modular or self-reconfiguring robot swarms specifically.

## Q4. DAgger and its 2024+ variants; handling stochastic or multimodal experts

### Takeaway
DAgger and interactive IL remain the standard fix for compounding error. 2024-2026 work mostly adapts DAgger to multimodal (diffusion) policies and to deciding *when* to query the expert. The key multimodality pitfall is that uncertainty-gated DAgger variants mistake legitimate multimodality (several equally valid actions) for uncertainty. No paper was found that directly addresses DAgger with a stochastic *rule* expert; the closest guidance is to model the expert's action distribution with a cross-entropy or distributional loss, or to use expressive policy classes.

### Cited Findings
**Foundational work (background, not re-fetched)**
- Ross, Gordon and Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger), AISTATS 2011. — [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)
- Kelly et al., "HG-DAgger: Interactive Imitation Learning with Human Experts," ICRA 2019. The human expert takes control only when needed. — [arXiv 1810.02890](https://arxiv.org/abs/1810.02890); [ResearchGate](https://www.researchgate.net/publication/335140291_HG-DAgger_Interactive_Imitation_Learning_with_Human_Experts)
- MEGA-DAgger extends DAgger to several imperfect experts. — [ResearchGate](https://www.researchgate.net/publication/368923248_MEGA-DAgger_Imitation_Learning_with_Multiple_Imperfect_Experts)

**2024-2026 work**
- Diff-DAgger (2024): robot-gated DAgger methods such as Ensemble-DAgger "can mistake disagreement between policies for uncertainty at multimodal decision points." Diff-DAgger instead uses diffusion-policy loss for uncertainty estimation. It reports better failure prediction and completion on stacking, pushing and plugging, and lower wall-clock cost. — [arXiv 2410.14868](https://arxiv.org/abs/2410.14868)
- DifNav (2025, vision-language navigation): prior work "largely overlooked the multi-modal nature of demonstrations generated during DAgger training, which can lead to inconsistent policies." DifNav pairs DAgger data collection with a diffusion policy. — [arXiv 2508.09444](https://arxiv.org/html/2508.09444v1)
- Policy-Calibrated DAgger (2026, arXiv): offline-calibrated noise injection for diffusion policies under DAgger; evaluated only on small simulated tasks. — [arXiv 2609.30462](https://arxiv.org/abs/2609.30462)
- Kim et al. (2025), query-when-needed DAgger-like IL for privileged teachers (Q1/Q2). — [arXiv 2505.09546](https://arxiv.org/abs/2505.09546)
- Swamy et al. (2022, background): DAgger-style on-policy data is what lets history-based students close the gap with unobserved contexts. — [arXiv 2208.02225](https://arxiv.org/abs/2208.02225)
- SCoRe (2025, LLM agents): the student rolls out and the teacher corrects only the earliest mistake. This is a 2025 DAgger-variant pattern in a different field. — [arXiv 2509.14257](https://arxiv.org/abs/2509.14257)

### Inferences
- For a discrete action space and a stochastic rule teacher, the simplest principled choice is to train on the teacher's full action distribution (soft labels), or on a uniform distribution over its tied-best actions, rather than one sampled action. This removes the label noise that caps agreement at around 65%, and makes "agreement" measurable as KL. Diffusion and other expressive policies are needed for multimodal *continuous* actions; a categorical softmax already represents multimodality over discrete moves. This is an inference consistent with the Diff-DAgger and DifNav motivation, not a cited lattice result.
- Two rounds of DAgger is modest. The literature does not prescribe a count. Gating DAgger queries on states where the student is off-distribution (Kim et al.; HG-DAgger style) is the 2024+ trend.
- Avoid ensemble-disagreement gating if the teacher is genuinely multimodal: Diff-DAgger's warning applies.

### Gaps
- No 2024+ paper was found that specifically studies DAgger with a *stochastic scripted or rule* expert, or quantifies agreement versus success under expert tie-breaking.
- Not checked: whether Diff-DAgger and DifNav have peer-reviewed venues.

## Q5. Symmetry: data augmentation vs equivariant networks, especially discrete lattice (cubic group)

### Takeaway
Consensus as of 2024-2025:
- Built-in equivariance is more *data-efficient*, especially in low-data regimes, and often generalizes better.
- Augmentation can close much of the gap given enough training epochs or compute, and in wide-network limits provably converges to group-convolutional behaviour.
- Exact equivariance hurts when the task's symmetry is only approximate or broken (gravity, a fixed compass frame), which motivates approximate or relaxed equivariance.

For a cubic lattice, the relevant group is discrete (rotations and mirrors of the cube). Discrete-group convolutions such as CubeNet and 3D G-CNN exist for voxels. No published lattice-robot or modular-robot policy using them was found.

### Cited Findings
**Augmentation vs equivariance (general ML)**
- Brehmer, Behrends, de Haan and Cohen, "Does equivariance matter at scale?," NeurIPS 2024 Workshop (NeurReps), arXiv 2410.23179, revised July 2025. Tested on a rigid-body interaction benchmark:
  - equivariant models are more data-efficient;
  - non-equivariant transformers *with augmentation* can close the gap given enough epochs;
  - equivariant models outperformed at every compute budget tested;
  - the optimal model-size/training-time split differs between the two.
  — [arXiv 2410.23179](https://arxiv.org/abs/2410.23179)
- Misof, Kessel and Gerken (2024): augmented networks converge to group-convolutional networks in the infinite-width limit, with approximate finite-width behaviour on CIFAR10/MNIST and ensemble results including OOD data. — [arXiv 2406.06504](https://arxiv.org/pdf/2406.06504)
- Nordenfors and Flinth (2024): ensembles provably learn equivariance through data augmentation (NTK limit). — [arXiv 2410.01452](https://arxiv.org/pdf/2410.01452)
- Biomedical imaging (Scientific Reports, 2024): a group-convolutional network without augmentation beat an augmented CNN, 82.43% vs 76.59% validation accuracy. — [Nature Sci. Rep. 2024](https://www.nature.com/articles/s41598-024-65597-x)
- Background (2022), spherical MNIST: equivariant models win at small data; augmented models keep improving with more data, and whether they catch up is unclear. — [arXiv 2202.03990](https://arxiv.org/pdf/2202.03990)
- Background (2022), dynamics forecasting: strictly equivariant networks have tighter generalization bounds than augmentation, and approximately equivariant models can beat both on imperfectly symmetric dynamics. — [arXiv 2206.09450](https://arxiv.org/pdf/2206.09450)
- 2025: "To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking." Augmentation and equivariance can discard useful symmetry-breaking information when the data distribution itself is not symmetric. — [arXiv 2510.01349](https://arxiv.org/html/2510.01349)

**Equivariant RL and MARL (2024-2025)**
- McClellan, Haghani, Winder, Huang and Tokekar, "Boosting Sample Efficiency and Generalization in Multi-agent RL via Equivariance," NeurIPS 2024.
  - EGNN policies improve MARL sample efficiency and generalization.
  - But "a naive application of EGNNs to MARL results in poor early exploration due to a bias in the EGNN structure." E2GN2 fixes this.
  - It reports a 2x-5x generalization gain over standard GNNs on MPE and SMACv2.
  — [arXiv 2410.02581](https://arxiv.org/abs/2410.02581)
- Park, Bhatt et al., "Approximate Equivariance in Reinforcement Learning" (2024). Many tasks are only approximately symmetric, so exact constraints can be inappropriate. Relaxed group convolutions improve sample efficiency and noise robustness. — [arXiv 2411.04225](https://arxiv.org/pdf/2411.04225)
- Equivariant RL under partial observability (2024). Its related-work section says augmentation-only training is typically less data-efficient, generalizes less, and needs larger models and longer training than equivariant designs. This is cited from prior work, not tested there. — [arXiv 2408.14336](https://arxiv.org/html/2408.14336v1)
- Equivariant RL for quadrotor low-level control (2025): equivariant variants beat non-equivariant ones in learning efficiency and flight performance. — [arXiv 2502.20500](https://arxiv.org/html/2502.20500v1)
- Equivariant action sampling for TD-MPC (WAFR 2024): learns faster and needs fewer samples than the non-equivariant version. — [Northeastern PDF](https://www.khoury.northeastern.edu/home/lsw/papers/wafr2024-sampling.pdf)
- Multi-Group Equivariant Augmentation for offline RL in manipulation (2025): augmentation used *as* the symmetry method, to raise sample efficiency. — [arXiv 2508.11204](https://arxiv.org/html/2508.11204)
- A 2025 RLJ/RLC paper on partial equivariance notes that imposed equivariance constraints may not reflect the true dynamics, leading to suboptimal performance. — [RLJ 2025 paper 310](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_310.pdf)

**Background, not re-fetched**
- van der Pol, Worrall, van Hoof, Oliehoek and Welling, "MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning," NeurIPS 2020: equivariant policy and value networks for discrete symmetries such as reflections and 90-degree rotations in grid-like RL. — [arXiv 2006.16908](https://arxiv.org/abs/2006.16908)
- Wang, Walters, Platt et al., "SO(2)-Equivariant Reinforcement Learning," ICLR 2022: the start of the 2022-2025 equivariant manipulation line, built on Equivariant SAC/DQN. — [arXiv 2203.04439](https://arxiv.org/abs/2203.04439)

**Discrete cubic group on voxels**
- CubeNet (Worrall and Brostow, ECCV 2018): a group-convolutional network equivariant to translations and right-angle (90-degree) rotations on voxels, described as the first 3D rotation-equivariant CNN for voxels. — [arXiv 1804.04458](https://ar5iv.labs.arxiv.org/html/1804.04458)
- 3D G-CNNs (Winkels and Cohen) are the other standard discrete-rotation voxel G-CNN, as cited by ILPO-NET. — [ILPO-NET, arXiv 2403.19612](https://arxiv.org/html/2403.19612v3)
- Limitation: 90-degree group convolutions cannot capture continuous rotation, and summing over rotations can produce radially symmetric filters that lose angular detail. — [ILPO-NET, arXiv 2403.19612](https://arxiv.org/html/2403.19612v3)

### Inferences
- **Which group applies.** For a cube lattice, the full symmetry group is the octahedral group O_h: 48 elements, 24 rotations times reflection. (This is general geometry, not from a fetched source.) Gravity and pivoting physics usually fix "up," which shrinks the valid group to the 8 symmetries that preserve the vertical axis: 4 rotations about z plus 4 vertical mirrors. The compass, target map and hop count must be transformed *consistently* with the window and the action labels, or augmentation teaches wrong labels.
- **Augmentation vs equivariance for this project.** With a small group (8 or 48 elements), cheap simulation and a teacher that can label unlimited data, the 2024 evidence (Brehmer et al.; the NTK results) suggests augmentation will recover most of the benefit of built-in equivariance, at modest extra training cost. Built-in equivariance matters more when data is scarce or labels are costly, which is not this setup's main bottleneck.
- **Lightweight exact alternative.** "Symmetrization" or test-time group averaging: run the MLP on all 8 (or 48) transformed inputs, untransform the action logits and average them. This gives exact equivariance with no architecture change. It is a standard technique, but was not separately sourced here.
- **Two cautions from the 2024-2025 literature:**
  - if the teacher or the world has asymmetries (gravity, a fixed compass frame, rule tie-breaking with a directional bias), forcing exact equivariance can hurt (Park et al. 2024; RLJ 2025; arXiv 2510.01349);
  - equivariant multi-agent networks can harm early exploration (E2GN2). That matters only if the student is later RL fine-tuned.
- **Interaction with the stochastic teacher.** Augmentation also averages out directional bias in the teacher's tie-breaking, which may raise agreement and lower variance.

### Gaps
- No 2024+ paper was found that directly compares augmentation and equivariant networks *in RL or IL* on a discrete 3D lattice or cubic group. The closest are the 2D grid MDP homomorphic networks (2020, background) and voxel G-CNNs for perception.
- No modular, self-reconfiguring or lattice-robot policy using octahedral-group equivariance was found.
- A direct benchmark of equivariant RL against strong augmentation baselines (such as DrQ-style) was not found; claims favouring equivariance in RL are mostly against non-symmetric baselines.
