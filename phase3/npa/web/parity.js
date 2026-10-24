/* Replay a JAX run through npa.js and report how far apart they end up.
 *
 *   node parity.js fixture.json weights.js
 *
 * fixture.json holds the starting positions and states, the update masks, and the JAX results
 * after 1 and 5 steps (checks.py writes it). weights.js is a file made by weights.to_js.
 * Prints one JSON line: the biggest position and state differences after 1 and 5 steps.
 */
const fs = require("fs");
const vm = require("vm");
const { Swarm } = require("./npa.js");

const [fixturePath, weightsPath] = process.argv.slice(2);
const win = {};
vm.runInNewContext(fs.readFileSync(weightsPath, "utf8"), { window: win });
const model = Object.values(win.NPA_MODELS)[0];
const f = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

const swarm = new Swarm(model, f.n, { eps: f.eps });
swarm.x.set(f.x);
swarm.s.set(f.s);
const gap = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

swarm.step(Uint8Array.from(f.masks[0]));
const after1 = { x: gap(swarm.x, f.x1), s: gap(swarm.s, f.s1) };
for (let k = 1; k < f.masks.length; k++) swarm.step(Uint8Array.from(f.masks[k]));
const after5 = { x: gap(swarm.x, f.x5), s: gap(swarm.s, f.s5) };
console.log(JSON.stringify({ after1, after5 }));
