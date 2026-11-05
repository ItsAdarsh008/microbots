# Lattice swarm under gravity, trained with no teacher

Side view of the `linking/` lattice: bots in grid cells, moves = stay, 4 slides along a surface, 4 pivots over an
edge (the bot keeps touching the structure). **Nothing is enforced about falling, supports or build order.**
Physics only: a bot is held if it has a column under it or sits within `L` cells (sideways or hanging) of a held
bot (cantilever limit; the v1 design note says one bot can stick out, `L = 1`). Unheld bots fall.

The controller is a small neural cellular automaton (three 3x3 convolutions: each bot sees a 7x7 window of
occupancy, the target blueprint, which neighbours are held and their hidden state; 22.5k weights), trained by
evolution strategies (`es.py`, antithetic, rank-shaped, Adam) on one number: the fraction of the target filled after
the build and a settling phase.

**Test for emergent scaffolding:** a perfect placement of the target alone keeps only 85% of its cells at L = 1
(validation shapes; 68% for the held-out mushrooms and trees): the overhangs fall. Filling more than that requires
bots outside the target holding the overhangs up, which nothing asks for.

    python lat/es.py lat/configs/<name>.json        # from emergent/; runs/<name>/{result.json, samples.png, traj.gif}
