# Microbot Sandbox

An in-browser 3D physics playground for the v1 microbot. The live copy is at https://claude.ai/artifact/36FggAsfw87ga8HjDYZVRv (private; share it from the page's Share menu).

| File | What it is |
| --- | --- |
| `sim.js` | Physics core (cannon-es): bots, joints, magnetic connectors, the break model and the preset scenes. It uses no DOM, so tests can run it in Node. |
| `index.src.html` | The page: Three.js rendering, controls and dragging |
| `build_page.py` | Inlines `sim.js` and the CAD STLs (`../stl/part_*.stl`) into `index.html` |
| `index.html` | The built single-file page (about 2 MB). Open it in a browser; it needs internet access for the CDN libraries. |

## How the physics works

- **Units and mass.** The sim uses real units (cm, g, s) and the CAD mass of 19.3 g per bot.
- **Rigid joints.** Each bot is one rigid body, and its collision shapes follow its joint angles. This models the design's self-locking joints: they hold any load at rest and move at 120°/s.
- **Connectors.** Facing tips or ports pull together magnetically. Within 3 mm and 30° they lock, at the nearest of the 24 tooth positions (every 15°).
- **Breaking.** The load on each link comes from the solver's constraint forces. A link breaks when tension exceeds the magnet's hold force (3 N by default), or when bending exceeds (hold + compression) × 5 mm.

**Checked headless:** one bot sticking out sideways holds at 7.7 mN·m (51% of its limit, the same as the design note's hand calculation), while two in a row break at the root. The hanging chain's top link carries 0.95 N, exactly the weight of the five bots below it.

Rebuild after editing: run `python ../build.py --fast` once to generate `../stl/`, then `python build_page.py`. The built `index.html` is gitignored because it embeds the meshes.
