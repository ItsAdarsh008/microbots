# Phase 2: from one lizard to any shape

Goal (not to be shrunk): a swarm model that **self-heals regardless of shape and topology**, in 2D. Getting there in steps,
each with a pass/fail test, none of which replaces the final goal:

**2a Multi-shape.** One network grows and heals several shapes (the seed carries a shape code in its first state
channels). Pass: each shape scores within 25% of a single-shape specialist trained with the same budget (`shape_<s>_base`
runs), and the architecture ranking from the lizard holds on at least 5 shapes.

**2b Shape-agnostic healing.** No target in the weights. The swarm starts as an arbitrary shape: particle positions are the
shape, and the colour channels hold the pattern. The network has to keep the swarm as it was, and bring it back after
damage. The reference is the swarm itself (rendered by `target.splat`), so *any* shape works, including ones never seen
in training. Train on many random and emoji shapes; evaluate on held-out emoji and on procedurally generated shapes with
holes, branches and thin parts. The floor to beat is a model that does nothing (zero update): it shows how much damage
costs with no repair. Pass: on held-out shapes the repair fraction (1 - (score - control) / (do_nothing - control)) is at
least 0.5 for wipe, shuffle and noise damage, and clearly above do-nothing for cut, drag and clump.

**2c Regrowth.** A cut deletes particles for good, and a rule that conserves N can only thin the survivors out (H11). Test two
mechanisms: continuous per-particle mass (growth and decay), and division into spare slots (mass 0 slots that come alive).
Pass: cut excess on the lizard and on held-out shapes falls to within 25% of the wipe excess.

Rules that do not change: updates stay local (neighbours within eps, nothing global, no coordinates); evaluators are
immutable and use held-out seeds and shapes; a result is real only if it is above the seed spread; report failures.
