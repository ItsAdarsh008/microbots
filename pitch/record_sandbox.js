// Plays the Linking Sandbox (linking/sandbox/index.html) headless and records every bot, every step, in the same
// layout as pitch/data/linking.js. Run through `python pitch/build_data.py sandbox`.
// Usage: node record_sandbox.js <sandbox index.html> <three@0.128.0 package dir> '<runs json>' <out.json>
// The sandbox comes from main (it has the 3x / 5x bots and parking); build_data.py pulls it with git show.
const { chromium } = require('playwright'); const fs = require('fs');
const [SB, T3] = process.argv.slice(2, 4).map(p => require('path').resolve(p));
const runs = JSON.parse(process.argv[4]), out = process.argv[5];
(async () => {
  const br = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const pg = await br.newPage();
  await pg.route('**/three.min.js', r => r.fulfill({ path: T3 + '/build/three.min.js', contentType: 'text/javascript' }));
  await pg.route('**/OrbitControls.js', r => r.fulfill({ path: T3 + '/examples/js/controls/OrbitControls.js', contentType: 'text/javascript' }));
  pg.on('pageerror', e => console.error('PAGE', e.message));
  await pg.goto('file://' + SB); await pg.waitForTimeout(1500);
  const res = {};
  for (const r of runs) {
    const d = await pg.evaluate(r => {
      playing = false;
      $("shape").value = r.shape; V1 = !!r.v1; $("v1").checked = V1; BOT_MULT = r.mult; BRAIN = "rules";
      reset();
      const B = S.pos.length, NB = [];
      const xyz = i => [i % NX, Math.floor(i / NX) % NY, Math.floor(i / (NX * NY))];
      const cell = [], flag = [];
      const snap = () => {
        for (let b = 0; b < B; b++) {
          const k = S.kind[b], to = S.pos[b];
          const fr = k ? S.from[b] : to, via = k ? S.via[b] : to;
          cell.push(...xyz(fr), ...xyz(to), ...xyz(via));
          const lk = S.alive[b] && S.locked[to] ? 1 : 0, sup = lk && S.support[to] ? 1 : 0;
          flag.push((k & 3) | (S.alive[b] ? 4 : 0) | (lk << 3) | (sup << 4));
        }
      };
      snap(); let t = 0, built = -1, quiet = 0;
      for (; t < r.max; t++) {
        tick(); snap();
        if (built < 0 && done()) built = t + 1;
        let moving = 0; for (let b = 0; b < B; b++) if (S.kind[b]) moving++;
        quiet = moving <= 1 ? quiet + 1 : 0;              // parking: stop once the spares have settled
        if (built >= 0 && (!r.park || parked() || quiet >= 12)) { t++; break; }
      }
      const ghost = []; for (let i = 0; i < N; i++) if (S.target[i]) ghost.push(...xyz(i), S.support[i] ? 1 : 0);
      const enc = a => { let s = ''; const u = Int8Array.from(a), v = new Uint8Array(u.buffer); for (let i = 0; i < v.length; i += 8192) s += String.fromCharCode.apply(null, v.subarray(i, i + 8192)); return btoa(s); };
      const encU = a => { let s = ''; const v = Uint8Array.from(a); for (let i = 0; i < v.length; i += 8192) s += String.fromCharCode.apply(null, v.subarray(i, i + 8192)); return btoa(s); };
      return { t: t + 1, b: B, grid: [NX, NY, NZ], built, need: S.need, cell: enc(cell), flag: encU(flag), target: enc(ghost) };
    }, r);
    console.log(r.name, 'bots', d.b, 'frames', d.t, 'built at', d.built, 'need', d.need, 'MB', ((d.cell.length + d.flag.length) / 1e6).toFixed(2));
    res[r.name] = d;
  }
  fs.writeFileSync(out, JSON.stringify(res)); await br.close();
})();
