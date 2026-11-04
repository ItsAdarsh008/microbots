# Emergent structure: train the swarm inside physics, do not write the behaviours

Direction (2026-11-04): stop replicating individual behaviours by hand (bending limits, connector rigidity rules,
support planners, repair rules). Give the swarm a physical world and an objective; whatever structural behaviour
is needed should come out of training.

## The question
Can one small controller, copied into every bot and seeing only its neighbours, learn to build a target shape
**that stands up under gravity**, when it is trained end to end through a simulated world in which:
- bots are discs of diameter 1 with mass 1, a floor, gravity, contact, Coulomb-like friction;
- a bot can only push or pull **against what it touches** (neighbours or floor; momentum is conserved, no flying);
- each bot chooses a grip; gripping pairs hold with a **latching bond of limited strength** (tension only, saturates
  at `f_bond`, breaks beyond 1.25 diameters);
- nothing else. **No bending stiffness, no rigidity rule, no support planner, no repair rule.** A single chain of
  central-force bonds has zero bending stiffness, so overhangs must find their own way to stay up.

## What would convince
1. The trained swarm's structure survives a long passive hold (motors off, grips kept) **better than the obvious
   arrangement**: an even sample of the target placed directly, all grips on ("ceiling-freeze"), in the same physics.
2. The difference is structural and visible: e.g. more bonds or triangles at the roots of overhangs, thicker
   members, counterweights, temporary supports, compared with ceiling-freeze and with the same controller trained
   without gravity.
3. It holds on held-out shape families.

**Falsified if:** the trained swarm never stands better than ceiling-freeze, or the gravity-trained and the
no-gravity-trained controllers make indistinguishable structures, or it only works on training families.

## References in every result
- `pile`: do nothing (grip on): chance level of the footprint IoU.
- `ceiling`: the even sample of the target, IoU at build time (upper reference) and after the passive hold.
- Controls: shuffled-target training (must fail), no-gravity training evaluated under gravity, external-force
  ("drone") actuation (unphysical upper bound on mobility).

## Metrics
Footprint IoU (bots as discs vs the target raster) at the end of the build and after the passive hold; cover
(fraction of target points with a bot within 0.75); sag (how far bots drop during the hold) and fallen fraction;
bonds per bot and bonded triangles per bot, split into overhang and supported regions. Pictures and GIFs of every run.

## Stages
- E1: 2D vertical plane, 128 bots, floor-standing shapes (towers, T/L cantilevers, arches, stairs, stacked blocks,
  trees; held out: mushrooms, Y shapes). Sweep bond strength, gravity, sensing radius, memory, seeds.
- E2: damage during the build or the hold (does repair emerge?), more bots, weaker friction.
- E3: 3D.
