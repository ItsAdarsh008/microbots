/* Neural Particle Automaton, in plain JavaScript.
 *
 * The same rule as phase3/npa/npa.py (a port of the paper's model, arXiv 2601.16096), but
 * neighbours come from a spatial hash grid instead of an N x N matrix, so thousands of
 * particles run in a browser. No DOM and no dependencies: it loads as a <script> (window.NPA)
 * and in Node (require), and checks.py runs it against the JAX version to prove they agree.
 *
 * A model is { eps, eps0, alpha, w1 (66x128), b1 (128), w2 (128x18) } as written by weights.to_js.
 * Each step, every particle senses its neighbours within eps (density, smoothed state, state
 * gradient, density gradient), runs them through a 2-layer MLP, and moves/updates itself.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.NPA = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const C = 16;                 // state channels per particle; the last 3 are the colour
  const D = 2;                  // space dimensions
  const P = C * (2 + D) + D;    // 66 numbers sensed per particle
  const H = 128;                // hidden units
  const O = D + C;              // 18 outputs: a move and a state change
  const UPDATE_PROB = 0.5;
  const DET_MIN = 1e-3;

  /** A swarm of n particles running one model. Positions are x[2i], x[2i+1]; states s[16i..]. */
  class Swarm {
    constructor(model, n, opts) {
      opts = opts || {};
      this.model = model;
      this.eps = opts.eps || model.eps;
      this.n = n;
      this.n0 = n;              // how many particles the swarm started with; reseed() brings them all back
      this.total = n;           // the count the density gradient is normalised by; cutting particles does not change it
      this.x = new Float32Array(n * D);
      this.s = new Float32Array(n * C);
      this.rng = opts.rng || Math.random;
      this.steps = 0;
      // scratch buffers, reused every step
      this.rho = new Float32Array(n);
      this.vol = new Float32Array(n);
      this.dx = new Float32Array(n * D);
      this.ds = new Float32Array(n * C);
      this.cell = new Int32Array(n);
      this.order = new Int32Array(n);
      this.cellStart = new Int32Array(1);
      this.mask = new Uint8Array(n);   // which particles update this step
      this.z = new Float32Array(P);
      this.h = new Float32Array(H);
      this.y = new Float32Array(O);
      this.reseed(opts.seedRadius);
    }

    /** A uniform disc of particles with zero state: the egg the shape grows from. */
    reseed(radius) {
      radius = radius || 2 * this.eps;     // the paper's seed is a disc of radius 2 * eps
      this.n = this.n0;
      for (let i = 0; i < this.n; i++) {
        const r = Math.sqrt(this.rng()) * radius, t = this.rng() * 2 * Math.PI;
        this.x[2 * i] = r * Math.cos(t);
        this.x[2 * i + 1] = r * Math.sin(t);
      }
      this.s.fill(0);
      this.steps = 0;
    }

    /** Delete every particle within r of (cx, cy), as if that piece were cut off. Returns how many went. */
    remove(cx, cy, r) {
      const r2 = r * r;
      let keep = 0;
      for (let i = 0; i < this.n; i++) {
        const ex = this.x[2 * i] - cx, ey = this.x[2 * i + 1] - cy;
        if (ex * ex + ey * ey < r2) continue;
        if (keep !== i) {
          this.x[2 * keep] = this.x[2 * i];
          this.x[2 * keep + 1] = this.x[2 * i + 1];
          this.s.copyWithin(keep * C, i * C, i * C + C);
        }
        keep++;
      }
      const gone = this.n - keep;
      this.n = keep;
      return gone;
    }

    /** Shift every particle within r of (cx, cy) by (dx, dy), carrying its state along. */
    drag(cx, cy, r, dx, dy) {
      const r2 = r * r;
      for (let i = 0; i < this.n; i++) {
        const ex = this.x[2 * i] - cx, ey = this.x[2 * i + 1] - cy;
        if (ex * ex + ey * ey < r2) { this.x[2 * i] += dx; this.x[2 * i + 1] += dy; }
      }
    }

    /** Landmarks on the grown lizard, found from the swarm itself: the particle furthest in a direction. */
    landmark(name) {
      const dir = { head: [-1, 1], tail: [1, -1], left: [-1, 0], right: [1, 0.3], top: [0, 1], bottom: [0, -1] }[name];
      let best = 0, bv = -Infinity;
      for (let i = 0; i < this.n; i++) {
        const v = dir[0] * this.x[2 * i] + dir[1] * this.x[2 * i + 1];
        if (v > bv) { bv = v; best = i; }
      }
      return [this.x[2 * best], this.x[2 * best + 1]];
    }

    /** Wipe the state of every particle within r of (cx, cy). Positions are untouched. */
    erase(cx, cy, r) {
      const r2 = r * r;
      for (let i = 0; i < this.n; i++) {
        const ex = this.x[2 * i] - cx, ey = this.x[2 * i + 1] - cy;
        if (ex * ex + ey * ey < r2) this.s.fill(0, i * C, i * C + C);
      }
    }

    /** Sort particles into a grid of eps-wide cells so each one only checks 9 cells. */
    _buildGrid() {
      const { n, x, eps } = this;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) {
        const px = x[2 * i], py = x[2 * i + 1];
        if (px < x0) x0 = px; if (px > x1) x1 = px;
        if (py < y0) y0 = py; if (py > y1) y1 = py;
      }
      this.gx0 = x0; this.gy0 = y0;
      this.gw = Math.min(Math.floor((x1 - x0) / eps) + 1, 1024);
      this.gh = Math.min(Math.floor((y1 - y0) / eps) + 1, 1024);
      const cells = this.gw * this.gh;
      if (this.cellStart.length !== cells + 1) this.cellStart = new Int32Array(cells + 1);
      else this.cellStart.fill(0);
      for (let i = 0; i < n; i++) {
        const cx = Math.min(Math.floor((x[2 * i] - x0) / eps), this.gw - 1);
        const cy = Math.min(Math.floor((x[2 * i + 1] - y0) / eps), this.gh - 1);
        const c = cy * this.gw + cx;
        this.cell[i] = c;
        this.cellStart[c + 1]++;
      }
      for (let c = 0; c < cells; c++) this.cellStart[c + 1] += this.cellStart[c];
      const fill = this.cellStart.slice(0, cells);
      for (let i = 0; i < n; i++) this.order[fill[this.cell[i]]++] = i;
    }

    /** Make room for at least `need` neighbour pairs. */
    _reservePairs(need) {
      if (this.pj && this.pj.length >= need) return;
      const cap = Math.max(need, this.pj ? this.pj.length * 2 : this.n * 80);
      const grow = (old, Type, k) => { const a = new Type(cap * k); if (old) a.set(old); return a; };
      this.pj = grow(this.pj, Int32Array, 1);
      this.pw = grow(this.pw, Float32Array, 1);    // smoothing kernel (poly6) of the pair
      this.pg = grow(this.pg, Float32Array, 2);    // gradient kernel (spiky) of the pair, as a vector
      this.pr = grow(this.pr, Float32Array, 2);    // offset from i to j
      this.pairStart = this.pairStart && this.pairStart.length >= this.n + 1 ? this.pairStart : new Int32Array(this.n + 1);
    }

    /**
     * Advance one step. `update` (optional, Uint8Array of 0/1) fixes which particles update,
     * so a test can replay the same randomness as the JAX version; by default each particle
     * updates with probability 0.5.
     */
    step(update) {
      const { n, x, s, eps, model, rho, vol, z, h, y } = this;
      const { w1, b1, w2 } = model;
      const eps2 = eps * eps;
      const polyNorm = 4 / (Math.PI * Math.pow(eps, 8));
      const spikyNorm = 10 / (Math.PI * Math.pow(eps, 5));
      const scale = eps / model.eps0;
      const dScale = Math.pow(scale, 1 + D) / this.total;
      this._buildGrid();
      const { gw, gh, cellStart, order, cell } = this;
      if (!this.pairStart || this.pairStart.length < n + 1) this.pairStart = new Int32Array(n + 1);
      this._reservePairs(n * 80);

      // pass 1: find every neighbour pair once, remember its kernels, and add up each density
      let m = 0;
      for (let i = 0; i < n; i++) {
        this.pairStart[i] = m;
        const xi = x[2 * i], yi = x[2 * i + 1];
        const c = cell[i], cx = c % gw, cy = (c - cx) / gw;
        let sum = 0;
        for (let yy = Math.max(cy - 1, 0); yy <= Math.min(cy + 1, gh - 1); yy++) {
          for (let xx = Math.max(cx - 1, 0); xx <= Math.min(cx + 1, gw - 1); xx++) {
            const cc = yy * gw + xx, end = cellStart[cc + 1];
            for (let k = cellStart[cc]; k < end; k++) {
              const j = order[k];
              const rx = x[2 * j] - xi, ry = x[2 * j + 1] - yi, d2 = rx * rx + ry * ry;
              if (d2 >= eps2) continue;
              if (m >= this.pj.length) this._reservePairs(m + 1);
              const t = eps2 - d2, w = polyNorm * t * t * t;
              sum += w;
              this.pj[m] = j; this.pw[m] = w; this.pr[2 * m] = rx; this.pr[2 * m + 1] = ry;
              if (d2 > 0) {                                  // the spiky gradient is zero at r = 0
                const d = Math.sqrt(d2), mag = spikyNorm * 3 * (eps - d) * (eps - d) / d;
                this.pg[2 * m] = mag * rx; this.pg[2 * m + 1] = mag * ry;
              } else { this.pg[2 * m] = 0; this.pg[2 * m + 1] = 0; }
              m++;
            }
          }
        }
        rho[i] = sum;
        vol[i] = 1 / sum;
      }
      this.pairStart[n] = m;
      const { pj, pw, pg, pr, pairStart } = this;

      // only a random half of the particles update each step, so only they need to sense and think
      const mask = this.mask;
      for (let i = 0; i < n; i++) mask[i] = update ? update[i] : (this.rng() < UPDATE_PROB ? 1 : 0);

      // pass 2: sense the neighbourhood and run the MLP
      const blur = new Float64Array(C), ax = new Float64Array(C), ay = new Float64Array(C);
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        blur.fill(0); ax.fill(0); ay.fill(0);
        let gvx = 0, gvy = 0, mxx = 0, mxy = 0, myx = 0, myy = 0, drx = 0, dry = 0;
        for (let p = pairStart[i], pe = pairStart[i + 1]; p < pe; p++) {
          const j = pj[p], v = vol[j], sj = j * C, wv = pw[p] * v;
          for (let c = 0; c < C; c++) blur[c] += wv * s[sj + c];
          if (j === i) continue;
          const gx = pg[2 * p], gy = pg[2 * p + 1], rx = pr[2 * p], ry = pr[2 * p + 1];
          const vgx = v * gx, vgy = v * gy;
          for (let c = 0; c < C; c++) { const sc = s[sj + c]; ax[c] += vgx * sc; ay[c] += vgy * sc; }
          gvx += vgx; gvy += vgy;
          mxx += rx * vgx; mxy += rx * vgy; myx += ry * vgx; myy += ry * vgy;
          drx += gx; dry += gy;
        }

        // first-order correction of the state gradient (skipped where the neighbours are too one-sided)
        const det = mxx * myy - mxy * myx;
        let i00 = 1, i01 = 0, i10 = 0, i11 = 1;
        if (det >= DET_MIN) { i00 = myy / det; i01 = -mxy / det; i10 = -myx / det; i11 = mxx / det; }

        const si = i * C;
        for (let c = 0; c < C; c++) {
          z[c] = s[si + c];
          z[C + c] = blur[c];
          const g0x = ax[c] - s[si + c] * gvx, g0y = ay[c] - s[si + c] * gvy;
          const gx = (g0x * i00 + g0y * i10) * scale, gy = (g0x * i01 + g0y * i11) * scale;
          const len = Math.sqrt(gx * gx + gy * gy + 1e-24), f = Math.log1p(len) / len;
          z[2 * C + 2 * c] = gx * f;
          z[2 * C + 2 * c + 1] = gy * f;
        }
        {
          const ex = drx * dScale, ey = dry * dScale, len = Math.sqrt(ex * ex + ey * ey + 1e-24);
          const f = Math.log1p(len) / len;
          z[P - 2] = ex * f; z[P - 1] = ey * f;
        }

        // MLP: relu(z @ w1 + b1) @ w2
        for (let k = 0; k < H; k++) h[k] = b1[k];
        for (let q = 0; q < P; q++) {
          const zq = z[q];
          if (zq === 0) continue;
          const row = q * H;
          for (let k = 0; k < H; k++) h[k] += zq * w1[row + k];
        }
        y.fill(0);
        for (let k = 0; k < H; k++) {
          const hk = h[k];
          if (hk <= 0) continue;
          const row = k * O;
          for (let o = 0; o < O; o++) y[o] += hk * w2[row + o];
        }
        const len = Math.sqrt(y[0] * y[0] + y[1] * y[1] + 1e-24);
        const mv = model.alpha * eps / (1 + len);
        this.dx[2 * i] = y[0] * mv;
        this.dx[2 * i + 1] = y[1] * mv;
        for (let c = 0; c < C; c++) this.ds[si + c] = y[D + c];
      }

      // apply the update, to a random half of the particles
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        x[2 * i] += this.dx[2 * i];
        x[2 * i + 1] += this.dx[2 * i + 1];
        for (let c = 0; c < C; c++) s[i * C + c] += this.ds[i * C + c];
      }
      this.steps++;
    }
  }

  /** The colour a particle shows, as 0-255 RGB: its last three state channels, centred on grey. */
  function colour(s, i, out) {
    for (let c = 0; c < 3; c++) {
      const v = s[i * C + C - 3 + c] + 0.5;
      out[c] = Math.round(255 * (v < 0 ? 0 : v > 1 ? 1 : v));
    }
    return out;
  }

  return { Swarm, colour, C, P, H, O };
});
