// Build every shape headlessly and report how it went.
//
//   cd linking/sandbox/tests
//   npm install && npx playwright install chromium
//   node run_shapes.js                       # all shapes, cubes then real render
//   node run_shapes.js bridge,table arms 1   # some shapes, one model, strength limit on
//
// For each run: bots, support bots, steps to finish, whether it finished,
// whether everything stayed attached at EVERY step, longest overhang, and
// milliseconds per step (slower than the page, since connectivity is
// re-checked every step here). three.js is served from node_modules so the
// test doesn't need the CDN.
const path = require("path");
const { chromium } = require("playwright");
const PAGE = "file://" + path.resolve(__dirname, "../index.html");
const NM = path.join(__dirname, "node_modules/three");

const shapes = (process.argv[2] || "bridge,arch,dome,pyramid,spiral,k,table,short").split(",");
const models = process.argv[3] ? [process.argv[3]] : ["faces", "arms"];
const strength = process.argv[4] === "1";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.route("**/three.min.js", r => r.fulfill({ path: path.join(NM, "build/three.min.js"), contentType: "application/javascript" }));
  await page.route("**/OrbitControls.js", r => r.fulfill({ path: path.join(NM, "examples/js/controls/OrbitControls.js"), contentType: "application/javascript" }));
  await page.route("**/fonts.googleapis.com/**", r => r.fulfill({ body: "", contentType: "text/css" }));
  const errors = []; page.on("pageerror", e => errors.push(String(e)));
  await page.goto(PAGE); await page.waitForTimeout(800);
  for (const model of models) for (const shape of shapes) {
    const r = await page.evaluate(([shape, model, strength]) => {
      playing = false; MODEL = model; V1 = strength;
      document.getElementById("shape").value = shape; reset();
      const t0 = performance.now(); let k = 0, attached = true;
      while (!done() && k < 1500) { tick(); k++; attached = attached && isConnected(); }
      let filled = 0; for (let i = 0; i < N; i++) if (S.target[i] && S.occ[i]) filled++;
      return { bots: S.pos.length, supports: S.nSupport, steps: k, finished: done(), filled: +(filled / S.need).toFixed(3),
               alwaysAttached: attached, msPerStep: +((performance.now() - t0) / Math.max(k, 1)).toFixed(1) };
    }, [shape, model, strength]);
    console.log(`${model.padEnd(5)} ${strength ? "strength" : "        "} ${shape.padEnd(8)} ${JSON.stringify(r)}`);
  }
  // the triangle truss bridge
  const t = await page.evaluate(() => {
    document.getElementById("shape").value = "truss"; pickShape(); playing = false;
    const out = [];
    for (let run = 0; run < 3; run++) {
      triReset(); let k = 0, attached = true;
      while (!tDone() && k < 800) { triTick(); k++; attached = attached && tAllAttached(); }
      out.push({ steps: k, finished: tDone(), alwaysAttached: attached });
    }
    return out;
  });
  console.log("truss            ", JSON.stringify(t));
  console.log(errors.length ? "page errors: " + errors.join(" | ") : "no page errors");
  await browser.close();
})();
