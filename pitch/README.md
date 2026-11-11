# Pitch deck (Emergent Ventures, 2 minutes)

A full-screen, wordless slideshow of the project. Open `index.html` in Chrome (double-click works; no server needed).

| Key | Does |
| --- | --- |
| click, tap, space, right arrow, PageDown | next step |
| left arrow, PageUp, right-click | back |
| F | full screen |
| 1 to 7 | jump to a slide |
| Home / End | first / last slide |

## Slides

1. One pile of 512 bots formed into shape after shape (the newest learned swarm).
2. Where it started: random dots, a flock from local rules, 150 bots assembling a K (Kilobot algorithm). 3 steps.
3. Microbot v1, the real CAD: one bot, then triangle, hexagon, 2x2x2 block. 4 steps.
4. 3D cube bots that stay linked: build an arch, then a slice is cut out and rebuilt. 2 steps.
5. The learned swarm (256 bots, arm-reach sensing): form a fish, cut a chunk, repair, then 24 shapes it never trained on. 4 steps.
6. The MVP we want to build: a flat round bot on three omni wheels, exploded view, then 40 of them running the policy. 3 steps.
7. The budget. The only slide with numbers.

Every animation is real project output, played back from `data/*.js` or simulated live in the page. The flat bot on
slide 6 is a concept drawn in code, not CAD.

## Changing things

- **Budget:** edit `BUDGET` at the top of the script in `index.html` (bot count, and one `[icon, dollars]` row per item). The total is the sum.
- **Data:** `python pitch/build_data.py swarm|kilobot|cad|linking` rebuilds `data/*.js`. `swarm` needs torch and the
  model weights (private Hugging Face repo, see `diffusion/WEIGHTS.md`); `kilobot` needs jax; `cad` reads the STL files
  from the `origin/meshes` branch.

## Live model demo

`python diffusion/demo/play.py --port 8001` serves a small page to play with the four newest swarm models (pick or draw
a shape, cut a chunk, repair). It needs torch, scipy and access to the weights.
