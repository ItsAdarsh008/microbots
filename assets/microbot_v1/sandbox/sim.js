// Microbot v1 physics core (cannon-es). No rendering and no DOM, so the page and
// the headless tests drive the same code. Units: cm, g, s. Forces are g·cm/s²
// (1 N = 1e5) and torques g·cm²/s² (1 mN·m = 1e4). Geometry and masses come from
// ../microbot.py and ../report.json.
//
// Each bot is ONE rigid body whose collision shapes follow its joint angles.
// That matches the hardware: the joints are self-locking worm drives that hold
// any load at rest and move at a set speed. So the physics is in the contacts,
// gravity and the magnetic connectors, and the solver never has to hold up a
// chain of tiny hinged parts.
//
// Bot frame = the CAD body frame: arms on ±Z poles, ports on ±X ±Y. The world is Y-up.

export function createSim(CANNON) {
  const { Vec3, Quaternion, Body } = CANNON;

  const G = {
    bodyR: 1.5, pinZ: 1.85, tipZ: 4.05, portD: 1.4, faceR: 0.5,
    mBot: 19.3, teeth: 24, pitchLimit: 100, jointSpeed: 120, // deg/s
  };

  // ---- transforms: {p: Vec3, q: Quaternion} ----------------------------------
  const X = (p = new Vec3(), q = new Quaternion()) => ({ p, q });
  const mul = (a, b) => X(a.p.vadd(a.q.vmult(b.p)), a.q.mult(b.q));
  const inv = (a) => { const qi = a.q.conjugate(); return X(qi.vmult(a.p).scale(-1), qi); };
  const qAxis = (ax, deg) => new Quaternion().setFromAxisAngle(ax, deg * Math.PI / 180);
  const T = (x, y, z) => X(new Vec3(x, y, z));
  const R = (ax, deg) => X(new Vec3(), qAxis(ax, deg));
  const EX = new Vec3(1, 0, 0), EY = new Vec3(0, 1, 0), EZ = new Vec3(0, 0, 1);
  const FLIP = R(EX, 180);
  // port base: local z -> bot +X, local x -> bot +Z (180 deg about (1,0,1)/sqrt2)
  const R0 = X(new Vec3(), new Quaternion(Math.SQRT1_2, 0, Math.SQRT1_2, 0));

  // ---- kinematics -----------------------------------------------------------------
  // pose (deg): ta/ba turret azimuth about bot Z, tp/bp arm pitch about the pin (X)
  const SIDES = { top: 1, bottom: -1 };
  function partFrames(pose) {
    const f = {};
    for (const [side, s] of Object.entries(SIDES)) {
      const turret = R(EZ, side === 'top' ? pose.ta : pose.ba);
      const pitch = side === 'top' ? pose.tp : pose.bp;
      const arm = mul(mul(mul(turret, T(0, 0, s * G.pinZ)), R(EX, pitch)), T(0, 0, -s * G.pinZ));
      f[side] = { turret, arm };
    }
    return f;
  }
  // tip connector frame in the arm frame (the arm frame equals the bot frame at zero pose)
  const tipInArm = { top: T(0, 0, G.tipZ), bottom: mul(T(0, 0, -G.tipZ), FLIP) };
  const PORTS = { '+X': 0, '+Y': 90, '-X': 180, '-Y': 270 };
  const portLocal = {};
  for (const [n, a] of Object.entries(PORTS)) portLocal[n] = mul(mul(R(EZ, a), T(G.portD, 0, 0)), R0);

  function connInBot(pose, name) {
    if (name in PORTS) return portLocal[name];
    return mul(partFrames(pose)[name].arm, tipInArm[name]);
  }
  function mateFrame(MA, poseB, cB, k = 0) {
    return mul(mul(mul(MA, FLIP), R(EZ, k * 360 / G.teeth)), inv(connInBot(poseB, cB)));
  }

  // collision shape frames for a pose (after the fixed body sphere)
  function shapeFrames(pose) {
    const f = partFrames(pose);
    const out = [];
    for (const [side, s] of Object.entries(SIDES)) {
      out.push(mul(f[side].turret, T(0, 0, s * 1.4)));                  // turret bar
      out.push(mul(f[side].arm, T(0, 0, s * G.pinZ)));                   // hub
      for (const z of [G.pinZ + 0.75, G.pinZ + 1.35, G.pinZ + 1.95]) out.push(mul(f[side].arm, T(0, 0, s * z)));
      out.push(mul(mul(f[side].arm, T(0, 0, s * (G.tipZ - 0.22))), R(EX, 90))); // tip disc (cylinder is along Y)
    }
    return out;
  }

  // ---- world ----------------------------------------------------------------------
  const world = new CANNON.World({ gravity: new Vec3(0, -981, 0) });
  world.broadphase = new CANNON.SAPBroadphase(world);
  world.solver.iterations = 30;
  world.allowSleep = false;
  world.defaultContactMaterial.friction = 0.6;
  world.defaultContactMaterial.restitution = 0.05;
  const ground = new Body({ mass: 0, shape: new CANNON.Plane() });
  ground.quaternion.setFromAxisAngle(EX, -Math.PI / 2);
  world.addBody(ground);

  const settings = {
    holdN: 3.0,         // EPM holding force per connector, N
    magnets: true,
    breakable: true,
    magnetRange: 1.5,   // cm
  };

  const bots = [];
  const links = [];
  const events = [];
  let nextId = 1, time = 0, breaks = 0;

  // ---- bots -------------------------------------------------------------------------
  function addBot(worldFrame, pose = { ta: 0, tp: 0, ba: 0, bp: 0 }, opts = {}) {
    const body = new Body({ mass: G.mBot, position: worldFrame.p.clone(), quaternion: worldFrame.q.clone() });
    body.linearDamping = 0.03; body.angularDamping = 0.08;
    body.addShape(new CANNON.Sphere(G.bodyR));
    const make = [
      () => new CANNON.Box(new Vec3(0.7, 0.33, 0.12)),
      () => new CANNON.Sphere(0.3),
      () => new CANNON.Sphere(0.4), () => new CANNON.Sphere(0.4), () => new CANNON.Sphere(0.4),
      () => new CANNON.Cylinder(0.5, 0.5, 0.44, 12),
    ];
    shapeFrames(pose).forEach((fr, i) => body.addShape(make[i % 6](), fr.p, fr.q));
    world.addBody(body);
    const bot = {
      id: nextId++, body, pose: { ...pose }, target: { ...pose }, pinned: false,
      conns: [], cooldown: 0, held: false,
    };
    body.bot = bot;
    for (const n of ['top', 'bottom', ...Object.keys(PORTS)]) bot.conns.push({ bot, name: n, link: null });
    bots.push(bot);
    if (opts.pinned) setPinned(bot, true);
    return bot;
  }

  function removeBot(bot) {
    for (const c of bot.conns) if (c.link) unlink(c.link, false);
    world.removeBody(bot.body);
    bots.splice(bots.indexOf(bot), 1);
  }
  function clear() { while (bots.length) removeBot(bots[0]); breaks = 0; events.length = 0; }

  function setPinned(bot, on) {
    const b = bot.body;
    bot.pinned = on;
    b.type = on ? Body.STATIC : Body.DYNAMIC;
    b.mass = on ? 0 : G.mBot;
    b.velocity.setZero(); b.angularVelocity.setZero();
    b.updateMassProperties();
    b.aabbNeedsUpdate = true;
  }

  const clampPitch = (v) => Math.max(-G.pitchLimit, Math.min(G.pitchLimit, v));
  const wrap = (d) => ((d % 360) + 540) % 360 - 180;
  function setPose(bot, pose) {
    for (const k of ['ta', 'tp', 'ba', 'bp']) if (pose[k] !== undefined) bot.target[k] = pose[k];
    bot.target.tp = clampPitch(bot.target.tp); bot.target.bp = clampPitch(bot.target.bp);
  }

  // move each joint toward its target at the joint speed, then re-pose the shapes
  function driveJoints(dt) {
    const maxStep = G.jointSpeed * dt;
    for (const bot of bots) {
      let moved = false;
      for (const k of ['ta', 'tp', 'ba', 'bp']) {
        const az = k.endsWith('a');
        const err = az ? wrap(bot.target[k] - bot.pose[k]) : bot.target[k] - bot.pose[k];
        if (Math.abs(err) < 1e-6) continue;
        bot.pose[k] += Math.max(-maxStep, Math.min(maxStep, err));
        if (az) bot.pose[k] = wrap(bot.pose[k]);
        moved = true;
      }
      if (!moved) continue;
      shapeFrames(bot.pose).forEach((fr, i) => {
        bot.body.shapeOffsets[i + 1].copy(fr.p);
        bot.body.shapeOrientations[i + 1].copy(fr.q);
      });
      bot.body.updateBoundingRadius();
      bot.body.aabbNeedsUpdate = true;
      for (const c of bot.conns) if (c.link) retarget(c.link);
    }
  }

  // ---- connectors ---------------------------------------------------------------------
  const connLocal = (c) => connInBot(c.bot.pose, c.name);
  function connWorld(c) { return mul(X(c.bot.body.position, c.bot.body.quaternion), connLocal(c)); }

  function idealRel(qA, qB, k) {
    return qA.mult(FLIP.q).mult(qAxis(EZ, k * 360 / G.teeth)).mult(qB.conjugate());
  }
  function quatAngle(a, b) {
    const d = Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
    return 2 * Math.acos(Math.min(1, d)) * 180 / Math.PI;
  }

  // point the lock's pivots and axes at the connectors' current local frames
  function retarget(L) {
    const la = connLocal(L.a), lb = connLocal(L.b);
    L.c.pivotA.copy(la.p); L.c.pivotB.copy(lb.p);
    const relInv = idealRel(la.q, lb.q, L.k).conjugate();
    L.c.xA.copy(EX); L.c.yA.copy(EY); L.c.zA.copy(EZ);
    L.c.xB.copy(relInv.vmult(EX)); L.c.yB.copy(relInv.vmult(EY)); L.c.zB.copy(relInv.vmult(EZ));
  }

  function link(cA, cB) {
    const A = cA.bot.body, B = cB.bot.body;
    const la = connLocal(cA), lb = connLocal(cB);
    const cur = A.quaternion.conjugate().mult(B.quaternion);
    let k = 0, bestAng = 1e9;
    for (let i = 0; i < G.teeth; i++) {
      const ang = quatAngle(idealRel(la.q, lb.q, i), cur);
      if (ang < bestAng) { bestAng = ang; k = i; }
    }
    if (bestAng > 30) return null;
    const c = new CANNON.LockConstraint(A, B, { maxForce: 1e12 });
    c.collideConnected = false;
    const L = { a: cA, b: cB, c, k, load: 0, tension: 0, bending: 0, age: 0 };
    retarget(L);
    world.addConstraint(c);
    cA.link = L; cB.link = L;
    links.push(L);
    return L;
  }

  function unlink(L, isBreak) {
    world.removeConstraint(L.c);
    L.a.link = null; L.b.link = null;
    links.splice(links.indexOf(L), 1);
    for (const c of [L.a, L.b]) c.bot.cooldown = Math.max(c.bot.cooldown, 0.8);
    if (isBreak) {
      breaks++;
      events.push({ t: time, kind: 'break', at: connWorld(L.a).p.clone(), load: L.load });
    }
  }
  function releaseBot(bot) {
    for (const c of bot.conns) if (c.link) unlink(c.link, false);
    bot.cooldown = 1.5;
  }

  function freeConns(ignoreCooldown = false) {
    const out = [];
    for (const bot of bots) {
      if (bot.cooldown > 0 && !ignoreCooldown) continue;
      for (const c of bot.conns) {
        if (c.link) continue;
        const w = connWorld(c);
        out.push({ c, p: w.p, n: w.q.vmult(EZ) });
      }
    }
    return out;
  }

  function magnetsAndSnap() {
    if (!settings.magnets) return;
    const free = freeConns();
    const F = settings.holdN * 1e5;
    const d3 = new Vec3();
    for (let i = 0; i < free.length; i++) {
      for (let j = i + 1; j < free.length; j++) {
        const A = free[i], B = free[j];
        if (A.c.bot === B.c.bot || A.c.link || B.c.link) continue;
        B.p.vsub(A.p, d3);
        const d = d3.length();
        if (d > settings.magnetRange) continue;
        const facing = -A.n.dot(B.n);
        if (facing < 0.3) continue;
        if (d < 0.3 && facing > 0.85 && link(A.c, B.c)) continue;
        // pull the faces together; strongest when close and square-on
        const s = F * 0.2 * facing * Math.pow(1 - d / settings.magnetRange, 2);
        const dir = d3.scale(1 / Math.max(d, 1e-3));
        const bA = A.c.bot.body, bB = B.c.bot.body;
        bA.applyForce(dir.scale(s), A.p.vsub(bA.position));
        bB.applyForce(dir.scale(-s), B.p.vsub(bB.position));
      }
    }
  }

  // ---- loads and the break model ---------------------------------------------------------
  // Force and torque through each link come from the solver's constraint multipliers.
  // A face fails in tension above the EPM hold force, and in bending when the
  // moment peels it open about its rim: M > (F_hold + compression) · r_face.
  let PULL_SIGN = -1; // checked by the headless test: a hanging chain must read as tension
  function measureLinks(dt) {
    const Fh = settings.holdN * 1e5;
    const a = 1 - Math.exp(-dt / 0.05); // 50 ms low-pass ignores one-step spikes
    for (const L of [...links]) {
      const c = L.c;
      const F = new Vec3(c.equationX.multiplier, c.equationY.multiplier, c.equationZ.multiplier);
      const Tq = new Vec3();
      for (const r of [c.rotationalEquation1, c.rotationalEquation2, c.rotationalEquation3]) {
        Tq.vadd(r.axisA.cross(r.axisB).scale(r.multiplier), Tq);
      }
      const n = connWorld(L.a).q.vmult(EZ);
      const axial = F.dot(n) * PULL_SIGN;             // > 0 pulls the faces apart
      const bend = Tq.vsub(n.scale(Tq.dot(n))).length();
      const inst = Math.max(Math.max(0, axial) / Fh, bend / ((Fh + Math.max(0, -axial)) * G.faceR));
      L.load += (inst - L.load) * a;
      L.tension += (axial / 1e5 - L.tension) * a;      // N
      L.bending += (bend / 1e4 - L.bending) * a;       // mN·m
      L.age += dt;
      if (settings.breakable && L.age > 0.3 && L.load > 1) unlink(L, true);
    }
  }

  // ---- dragging --------------------------------------------------------------------------
  const drag = { bot: null, local: new Vec3(), target: new Vec3() };
  function startDrag(bot, worldPoint) {
    drag.bot = bot; bot.held = true;
    bot.body.pointToLocalFrame(worldPoint, drag.local);
    drag.target.copy(worldPoint);
  }
  function moveDrag(p) { drag.target.copy(p); }
  function endDrag() { if (drag.bot) drag.bot.held = false; drag.bot = null; }
  function applyDrag() {
    const bot = drag.bot;
    if (!bot || bot.pinned) return;
    const b = bot.body;
    const p = b.pointToWorldFrame(drag.local);
    const v = b.getVelocityAtWorldPoint(p, new Vec3());
    const m = G.mBot, w = 24;
    const f = drag.target.vsub(p).scale(m * w * w).vsub(v.scale(2 * w * m));
    f.vsub(world.gravity.scale(m), f);              // carry the held bot's own weight
    b.applyForce(f, p.vsub(b.position));
    b.angularVelocity.scale(0.97, b.angularVelocity);
  }

  // ---- stepping ----------------------------------------------------------------------------
  const DT = 1 / 240;
  function step(substeps = 4) {
    for (let i = 0; i < substeps; i++) {
      driveJoints(DT);
      magnetsAndSnap();
      applyDrag();
      world.step(DT);
      measureLinks(DT);
      for (const bot of bots) bot.cooldown = Math.max(0, bot.cooldown - DT);
      time += DT;
    }
  }

  // ---- poses and scenes --------------------------------------------------------------------
  const vPose = (inter) => ({ ta: 0, tp: -(180 - inter) / 2, ba: 0, bp: (180 - inter) / 2 });
  const POSES = {
    straight: { ta: 0, tp: 0, ba: 0, bp: 0 },
    V120: vPose(120), V90: vPose(90), V60: vPose(60),
    U: { ta: 0, tp: -100, ba: 0, bp: 100 },
    S: { ta: 0, tp: -60, ba: 0, bp: -60 },
    twist: { ta: 90, tp: -60, ba: 0, bp: 60 },
  };

  const LAY_FLAT = X(new Vec3(), qAxis(EZ, 90));   // bot X -> world up
  const STAND = X(new Vec3(), qAxis(EX, -90));      // bot Z -> world up
  const SIDEWAYS = X(new Vec3(), qAxis(EY, 90));    // bot Z -> world +X

  function chainPlan(start, pose, n, k) {
    const frames = [start];
    for (let i = 1; i < n; i++) frames.push(mateFrame(mul(frames[i - 1], connInBot(pose, 'bottom')), pose, 'top', k));
    return frames;
  }
  function spawnPlan(frames, pose, pinnedIdx = []) {
    const made = frames.map((f, i) => addBot(f, pose, { pinned: pinnedIdx.includes(i) }));
    autoLink();
    return made;
  }
  // link every pair of connectors that coincide, e.g. right after a plan is spawned
  function autoLink() {
    const free = freeConns(true);
    for (let i = 0; i < free.length; i++) for (let j = i + 1; j < free.length; j++) {
      const A = free[i], B = free[j];
      if (A.c.bot === B.c.bot || A.c.link || B.c.link) continue;
      if (A.p.distanceTo(B.p) < 0.05 && A.n.dot(B.n) < -0.99) link(A.c, B.c);
    }
  }

  function ring(n) {
    const pose = vPose(180 - 360 / n);
    const plan = chainPlan(X(), pose, n, 0);
    const flat = plan.map((f) => mul(LAY_FLAT, f));
    let cx = 0, cz = 0, minY = Infinity;
    for (const f of flat) { cx += f.p.x / n; cz += f.p.z / n; minY = Math.min(minY, f.p.y); }
    spawnPlan(flat.map((f) => mul(T(-cx, -minY + G.bodyR + 0.05, -cz), f)), pose);
    return { pose, plan };
  }

  const SCENES = {
    one() { spawnPlan([mul(T(0, G.bodyR + 0.05, 0), LAY_FLAT)], POSES.V120); },
    ring3() { ring(3); },
    ring4() { ring(4); },
    ring6() { ring(6); },
    hanging() {
      const n = 6;
      spawnPlan(chainPlan(mul(T(0, 2 * G.tipZ * n + 4, 0), STAND), POSES.straight, n, 0), POSES.straight, [0]);
    },
    cantilever() {
      // back: one bot sticking out sideways (holds); front: two (the root peels open)
      for (const [z, n] of [[-7, 2], [7, 3]]) {
        spawnPlan(chainPlan(mul(T(8, 14, z), SIDEWAYS), POSES.straight, n, 0), POSES.straight, [0]);
      }
    },
    arch() {
      // 6 bots from a 10-sided ring, standing up with the chord level; ends pinned
      const pose = vPose(180 - 36);
      const plan = chainPlan(X(), pose, 6, 0);
      const d = plan[5].p.vsub(plan[0].p);
      const phi = Math.atan2(d.y, d.z) * 180 / Math.PI;
      let placed = plan.map((f) => mul(R(EX, phi), f));
      if (placed[2].p.y < placed[0].p.y) placed = plan.map((f) => mul(R(EX, phi + 180), f));
      let minY = Infinity, cz = 0;
      for (const f of placed) { minY = Math.min(minY, f.p.y); cz += f.p.z / placed.length; }
      spawnPlan(placed.map((f) => mul(T(0, -minY + G.tipZ + 0.6, -cz), f)), pose, [0, placed.length - 1]);
    },
    block() {
      const pose = POSES.straight;
      const frames = {};
      for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
        let f;
        if (!i && !j && !k) f = mul(T(0, G.tipZ + 0.3, 0), STAND);
        else if (i) f = mateFrame(mul(frames[`${i - 1}${j}${k}`], portLocal['+X']), pose, '-X');
        else if (j) f = mateFrame(mul(frames[`${i}${j - 1}${k}`], portLocal['+Y']), pose, '-Y');
        else f = mateFrame(mul(frames[`${i}${j}${k - 1}`], connInBot(pose, 'top')), pose, 'bottom');
        frames[`${i}${j}${k}`] = f;
      }
      const fs = Object.values(frames);
      let cx = 0, cz = 0;
      for (const f of fs) { cx += f.p.x / fs.length; cz += f.p.z / fs.length; }
      spawnPlan(fs.map((f) => mul(T(-cx, 0, -cz), f)), pose);
    },
    pile() {
      const keys = Object.keys(POSES);
      for (let i = 0; i < 12; i++) {
        const q = new Quaternion().setFromEuler(Math.random() * 6.3, Math.random() * 6.3, Math.random() * 6.3);
        addBot(X(new Vec3((Math.random() - 0.5) * 7, 6 + i * 4.5, (Math.random() - 0.5) * 7), q), POSES[keys[i % keys.length]]);
      }
    },
  };

  function loadScene(name) { clear(); SCENES[name](); }

  function dropBot(poseName = 'V120') {
    const q = new Quaternion().setFromEuler(Math.random() * 6.3, Math.random() * 6.3, Math.random() * 6.3);
    let top = 0;
    for (const b of bots) top = Math.max(top, b.body.position.y);
    return addBot(X(new Vec3((Math.random() - 0.5) * 6, top + 10, (Math.random() - 0.5) * 6), q), POSES[poseName]);
  }

  function stats() {
    let worst = 0;
    for (const L of links) worst = Math.max(worst, L.load);
    return { bots: bots.length, links: links.length, breaks, worst, time };
  }

  return {
    CANNON, G, world, settings, bots, links, events, POSES, SCENES,
    addBot, removeBot, clear, setPinned, setPose, releaseBot, dropBot, loadScene, step,
    startDrag, moveDrag, endDrag, drag, stats, connWorld, connInBot, partFrames,
    tipInArm, portLocal, mateFrame, chainPlan, ring, mul, inv, X, T, R,
    _setPullSign: (v) => { PULL_SIGN = v; },
  };
}
