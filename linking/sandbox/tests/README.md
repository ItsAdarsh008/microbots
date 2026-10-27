# Sandbox tests

Headless runs of `../index.html` that produced the numbers in `../../EXPERIMENTS.md`.

```bash
npm install
npx playwright install chromium
node run_shapes.js                       # every shape, cube model then arm model
node run_shapes.js bridge,table arms 1   # chosen shapes, arm model, strength limit on
```

Each line: bots, support bots, steps to finish, whether it finished, whether everything stayed attached at every step, and ms per step.
