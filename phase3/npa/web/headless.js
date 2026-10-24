/* Run the demo's engine without a browser and save snapshots of the swarm.
 *
 *   node headless.js weights.js out_dir [--particles 4096] [--steps 2000] [--every 100] [--seed 0]
 *                    [--events '[{"at":200,"kind":"erase","where":"head","r":0.3}]']
 *
 * Writes out_dir/frames.bin (for each snapshot: n x 2 positions, then n x 16 states, as float32)
 * and out_dir/meta.json (which lists n for every snapshot, since cutting changes it).
 * longrun.py and animate.py turn these into curves, figures and GIFs.
 *
 * An event damages the swarm when it reaches step `at`:
 *   erase  wipe the state of particles within r of a landmark (what training damage does)
 *   cut    delete the particles within r of a landmark, as if that piece were snipped off
 *   drag   shift the particles within r of a landmark by (dx, dy), tearing them off the body
 * `where` is a landmark found on the swarm itself: head, tail, left, right, top, bottom.
 * A snapshot tagged "damaged" is saved right after each event.
 */
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const { Swarm } = require("./npa.js");

const args = process.argv.slice(2);
const [weightsPath, outDir] = args;
const flag = (name, fallback) => { const i = args.indexOf("--" + name); return i < 0 ? fallback : args[i + 1]; };
const n = Number(flag("particles", 4096)), steps = Number(flag("steps", 2000));
const every = Number(flag("every", 100)), seed = Number(flag("seed", 0));
const events = JSON.parse(flag("events", "[]"));

// a small seeded generator, so a run can be repeated
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const win = {};
vm.runInNewContext(fs.readFileSync(weightsPath, "utf8"), { window: win });
const model = Object.values(win.NPA_MODELS)[0];
const swarm = new Swarm(model, n, { eps: model.eps, rng: mulberry32(seed) });

fs.mkdirSync(outDir, { recursive: true });
const fd = fs.openSync(path.join(outDir, "frames.bin"), "w");
const saved = [];
const snapshot = (tag, mark) => {
  const x = swarm.x.subarray(0, swarm.n * 2), s = swarm.s.subarray(0, swarm.n * 16);
  fs.writeSync(fd, Buffer.from(x.buffer, x.byteOffset, x.byteLength));
  fs.writeSync(fd, Buffer.from(s.buffer, s.byteOffset, s.byteLength));
  saved.push({ step: swarm.steps, tag, n: swarm.n, mark: mark || null });
};

function apply(ev) {
  const [cx, cy] = swarm.landmark(ev.where);
  let removed = 0;
  if (ev.kind === "erase") swarm.erase(cx, cy, ev.r);
  else if (ev.kind === "cut") removed = swarm.remove(cx, cy, ev.r);
  else if (ev.kind === "drag") swarm.drag(cx, cy, ev.r, ev.dx, ev.dy);
  else throw new Error("unknown event kind " + ev.kind);
  snapshot("damaged", [cx, cy, ev.r]);
  return removed;
}

const t0 = Date.now();
snapshot("start");
for (let t = 1; t <= steps; t++) {
  for (const ev of events) if (ev.at === t) ev.removed = apply(ev);
  swarm.step();
  if (t % every === 0) snapshot("run");
}
fs.closeSync(fd);
const seconds = (Date.now() - t0) / 1000;
fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify({ n, eps: swarm.eps, steps, seed, events, seconds, frames: saved }));
console.log(`${steps} steps of ${n} particles in ${seconds.toFixed(1)} s (${(steps / seconds).toFixed(1)} steps/s)`);
