# Robot hypotheses (from lab/ROBOT_RESEARCH.md; status updated as tests finish)

R-numbers follow ROBOT_RESEARCH.md section 3. Status: open / running / supported / refuted. Single runs are not proof.

| ID | Hypothesis (short) | Test | Status |
| --- | --- | --- | --- |
| R1 | "Returns to the shape" does not imply "strong": repaired swarms score poorly on rigidity vs a packed reference | `lab/r1_audit.py` on final configurations, metrics from `lab/strength.py` | running (first audit launched 22:10) |
| R2 | Triangular packing is much stronger than the square grid four 90-degree ports encourage; arm-tip diagonals brace a square | particle graph + MuJoCo 2x3 patches | open |
| R3 | Connector moment stiffness has a sharp threshold between pin-like and welded behaviour; the abstract metric ranks MuJoCo configs the same way | stiffness sweep + MuJoCo chain vs triangle | open |
| R4 | A tip-docked 3-bot ring is the smallest rigid cell even with pin-like connectors | MuJoCo load test | open |
| R5 | The failed docked-pair turn was a setup problem: dock on the roll axis through a free hinge with the docking arm idle | `dock_pair.py` with a hinge dock, 5 repeats | open |
| R6 | A rigidly docked cluster of one-line bots moves only if all roll lines agree: move undocked, dock last, melt only the damaged edge | MuJoCo welded pairs, orthogonal vs parallel poles | open |
| R7 | (bold) "Dock to two non-collinear docked neighbours" grows a rigid structure with no global check | particle docking simulation | open |
| R8 | Greedy docking by rigidity gain needs fewer bonds than docking everything touching | pebble-game comparison | open |
| R9 | Local features (triangle count, degree) predict critical bonds | logistic regression on random graphs | open |
| R10 | (bold) A diffusing stress / Fiedler channel lets the swarm find and thicken a weak neck | dumbbell shape, hand rule | open |
| R11 | A differentiable worst-mode-stiffness loss raises strength at small fidelity cost | add loss to the trainer, n <= 150 | open |
| R12 | (bold) A compass-free swarm can rotate a whole shape by projecting a tangential command onto each roll line | hand rule on a 300-bot disc | open |
| R13 | Reciprocal bearings give relative axis headings; doubled-angle consensus yields a common axis frame | consensus under noise and loss | open |
| R14 | Two-stage docking approach (coarse roll, then capture) beats single-stage with measured stride noise | particle + MuJoCo | open |
| R15 | Arm servos cannot carry structural load; strength must come from connectors and locked struts | MuJoCo tip-hold load test | open |
| R16 | Redundant rigidity predicts survival of 5-15% bot failure better than degree | random graphs | open |
| R17 | A docked bond is rigid until a load threshold, so strength is a threshold, not a stiffness | per-bond break rule load ramp | open |
