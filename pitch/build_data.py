"""Builds the data files the pitch deck plays back (pitch/data/*.js). Every animation in the deck is real output:
    python pitch/build_data.py swarm      # flow-matching swarm runs (needs torch, scipy; e.g. the mini-vla env)
    python pitch/build_data.py kilobot    # Phase 2 Kilobot-style self-assembly (needs jax)
    python pitch/build_data.py cad        # microbot v1 meshes from the origin/meshes branch (needs numpy, scipy)
    python pitch/build_data.py linking    # 3D linked cube bots: arch built, cut, repaired (needs numpy)
Files are plain <script> data (window.DATA.x = ...), so the deck opens from disk with no server."""
import base64, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")
Q = 20000.0                                   # positions stored as int16 = round(x * Q)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def write(name, obj):
    p = os.path.join(OUT, name + ".js")
    with open(p, "w") as f:
        f.write(f"(window.DATA=window.DATA||{{}}).{name}=" + json.dumps(obj, separators=(",", ":")) + ";\n")
    print(f"{name}.js  {os.path.getsize(p) / 1e6:.2f} MB")


def frames_q(frames):
    return b64(np.clip(np.round(np.asarray(frames, np.float32) * Q), -32767, 32767).astype("<i2"))


def occ_bits(occ):
    return "".join("1" if v > 0.5 else "0" for v in np.asarray(occ).reshape(-1))


# ---------------------------------------------------------------------------------------------------------------
def swarm():
    sys.path.insert(0, os.path.join(ROOT, "diffusion", "demo"))
    import engine as E
    L = E.shape_library()
    WIDE, ARM = "s14_refv3_r40_g16", "s19_refv3_r14_dmg08_big"

    def run(model, shape, N, nf=49, **kw):
        occ, sdf = L[shape]
        r = E.run_model(model, occ, sdf, N=N, K=64, seed=0, keep_spacings=True, max_frames=nf, **kw)
        return r, occ

    def pack(r, occ, **extra):
        f = np.asarray(r["frames"])
        return dict(n=int(f.shape[1]), f=int(f.shape[0]), sp=round(float(r["sp"]), 4), occ=occ_bits(occ), xy=frames_q(f),
                    iou=round(r["iou"], 3), **extra)

    # 1. hero: one pile of 512 bots, formed into shape after shape (each run starts from the same pile)
    hero = []
    for s in ["emoji star", "emoji butterfly", "letter K (thick)", "emoji fish", "emoji key"]:
        r, occ = run(WIDE, s, 512); hero.append(pack(r, occ, shape=s)); print("hero", s, round(r["iou"], 2))

    # 2. cut and repair at the arm-reach sensing range (the model trained on big cuts)
    rep = []
    for s, c in [("emoji fish", (-0.35, 0.05))]:
        r, occ = run(ARM, s, 256)
        X = r["final"]; keep = np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) > 0.34
        r2 = E.run_model(ARM, occ, L[s][1], K=64, seed=0, keep_spacings=True, n_ref=256, x0=X[keep], t0=0.0, N=int(keep.sum()), max_frames=49)
        rep.append(dict(form=pack(r, occ, shape=s), keep=b64(keep.astype(np.uint8)), cut=[c[0], c[1], 0.34],
                        repair=pack(r2, occ), cover=[round(float(E.extra(X[keep], occ, L[s][1])["cover"]), 3), round(r2["cover"], 3)]))
        print("repair", s, "removed", int((~keep).sum()), "cover", rep[-1]["cover"])

    # 3. wall: 24 shapes the models never trained on, all at once: the 12 emoji, then the first 12 held-out family
    #    shapes that read at thumbnail size (overlap >= 0.7; the thin "spider" shapes are the known weak spot and are skipped)
    wall = []
    for s in [n for n in L if n.startswith("emoji ")]:
        r, occ = run(WIDE, s, 256, nf=33); wall.append(pack(r, occ, shape=s)); print("wall", s, round(r["iou"], 2))
    for i in range(0, 200, 3):
        s = f"held-out #{i}"; r, occ = run(WIDE, s, 256, nf=33); print("wall", s, round(r["iou"], 2), "" if r["iou"] >= 0.7 else "skipped")
        if r["iou"] >= 0.7:
            wall.append(pack(r, occ, shape=s))
        if len(wall) == 24:
            break

    # 4. pucks: the same policy at MVP scale (40 bots, hard-core spacing rule on, as real bodies need)
    pucks = []
    for s in ["emoji star", "letter K (thick)", "emoji fish"]:
        r, occ = run(WIDE, s, 40, hardcore=True); pucks.append(pack(r, occ, shape=s)); print("pucks", s, round(r["iou"], 2))
    write("swarm", dict(q=Q, hero=hero, repair=rep, wall=wall, pucks=pucks))


# ---------------------------------------------------------------------------------------------------------------
def kilobot():
    sys.path.insert(0, os.path.join(ROOT, "phase2"))
    import assemble
    mask, pos, state, _, is_seed = assemble.run("k", 150, 7000, 20)
    pos, state, is_seed, mask = np.asarray(pos), np.asarray(state), np.asarray(is_seed), np.asarray(mask)
    pos, state = pos[::2], state[::2]
    ext = [float(assemble.ORIGIN[0]), float(assemble.ORIGIN[1]), float(assemble.GRID_SIZE * assemble.RES)]
    lo, hi = pos.min(), pos.max(); print("kilobot frames", pos.shape, "range", lo, hi, "extent", ext, "mask", mask.shape)
    s = float(30000.0 / max(abs(lo), abs(hi)))
    write("kilobot", dict(n=int(pos.shape[1]), f=int(pos.shape[0]), s=s, xy=b64(np.round(pos * s).astype("<i2")),
                          state=b64(state.astype(np.uint8)), seed=is_seed.astype(int).tolist(), ext=ext,
                          g=int(mask.shape[0]), mask="".join("1" if v else "0" for v in mask.T.reshape(-1))))


# ---------------------------------------------------------------------------------------------------------------
def cad():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def stl(name):
        raw = subprocess.run(["git", "-C", ROOT, "show", f"origin/meshes:microbot_v1/stl/{name}.stl"], capture_output=True, check=True).stdout
        n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
        rec = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
        return rec["v"].copy()                                     # (T, 3, 3)

    PARTS = ["body", "turret", "arm", "tip", "pin"]
    count = {p: len(stl("part_" + p)) for p in PARTS}; print("part triangle counts", count)

    def convert(name):
        V = stl(name); T = len(V)
        uniq, inv = np.unique(V.reshape(-1, 3), axis=0, return_inverse=True)
        tri = inv.reshape(T, 3)
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]]])
        nc, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(uniq),) * 2), directed=False)
        tl = lab[tri[:, 0]]; sizes = np.bincount(tl, minlength=nc)
        kind = np.array([min(range(5), key=lambda k: abs(count[PARTS[k]] - s)) for s in sizes])
        bad = [(int(s), PARTS[k]) for s, k in zip(sizes, kind) if s != count[PARTS[k]]]
        tk = kind[tl]; order = np.argsort(tk, kind="stable"); tri = tri[order]
        ranges = [int((tk == k).sum()) * 3 for k in range(5)]
        c = (uniq.min(0) + uniq.max(0)) / 2; P = uniq - c; r = float(np.abs(P).max())
        print(f"{name}: {T} tris, {len(uniq)} verts, {nc} solids, radius {r:.1f} mm" + (f"  UNMATCHED {bad[:4]}" if bad else ""))
        return dict(nv=len(uniq), r=r, pos=b64(np.round(P / r * 32767).astype("<i2")), idx=b64(tri.astype("<u4")), ranges=ranges)

    write("cad", {n: convert(n) for n in ["bot_straight", "bot_V120", "bot_U_grip", "ring3_triangle", "ring6_hexagon", "block_2x2x2"]})


# ---------------------------------------------------------------------------------------------------------------
def linking():
    sys.path.insert(0, os.path.join(ROOT, "linking"))
    import lattice as L
    target, frames, swarm = L.run("arch", seed=0, damage_region=L.cut_box(14, 18, 0, 16, 0, 14), extra=30)
    T, B = len(frames), len(frames[0]["pos"])
    cell = np.zeros((T, B, 9), np.int8); flag = np.zeros((T, B), np.uint8)       # from, pos, via ; kind | alive<<2 | locked<<3
    for t, f in enumerate(frames):
        cell[t, :, 0:3], cell[t, :, 3:6], cell[t, :, 6:9] = np.asarray(f["from"]), np.asarray(f["pos"]), np.asarray(f["via"])
        flag[t] = np.asarray(f["kind"]).astype(np.uint8) | (np.asarray(f["alive"]).astype(np.uint8) << 2) | (np.asarray(f["bot_locked"]).astype(np.uint8) << 3)
    dmg = next((i for i, f in enumerate(frames) if f["damaged"]), -1)
    done = next((i for i, f in enumerate(frames) if f["filled"] >= 1.0), T)
    print(f"linking: {T} steps, {B} bots, damaged at {dmg}, first built at {done}, always connected: {all(f['connected'] for f in frames)}")
    write("linking", dict(t=T, b=B, grid=list(map(int, L.GRID)), dmg=int(dmg), built=int(done), cell=b64(cell), flag=b64(flag),
                          target=np.argwhere(target).astype(int).tolist()))


if __name__ == "__main__":
    for a in sys.argv[1:] or ["swarm", "kilobot", "cad", "linking"]:
        dict(swarm=swarm, kilobot=kilobot, cad=cad, linking=linking)[a]()
