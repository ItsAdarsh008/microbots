"""Build gallery/index.html from gallery/*.json + *.gif (made by lat/gallery.py on the boxes).
Verdicts and notes are written by hand below (they are judgements, checked against the journal)."""
import html
import json
import os
import sys

G = sys.argv[1] if len(sys.argv) > 1 else "gallery"
BASE = dict(L=1, spares=0.3, C=8, M=32, pop=128, sigma=0.04, T=160, temp=1.0, families=None, damage=0.0, switch=0.0,
            move_w=0.0, kick=0.0, obj="fill", cure=0, cure_below=False, clean_w=0.0, clean_ramp=0, init_from=None,
            chasm=False, seed=0, H=20, W=40)
LAB = dict(L="cantilever limit L", spares="spare bots", C="memory channels", M="network width", pop="ES population",
           sigma="ES noise", T="build steps", temp="action temperature", families="trained on", damage="damage",
           switch="target switch", move_w="move cost", kick="kick after build", obj="objective", cure="cure after",
           cure_below="cures only on formwork", clean_w="clean-site cost", clean_ramp="cost ramp", init_from="warm start",
           chasm="chasm world", seed="seed", H="height", W="width")

# group, title, verdict (works / partial / fails / control / running / inconclusive), note
V = {
 "untrained": ("ref", "Untrained controller", "control", "Uniform random over the allowed moves. Fills about 2%; nothing is propped."),
 "perfect": ("ref", "Perfect placement, then gravity", "control", "The target alone at L = 1: overhangs beyond one cell fall (orange rubble). A swarm that beats this fill must be propping."),
 "lat1_L1": ("prop", "Base: L = 1, all shapes", "works", "Beats the target-alone reference on training shapes (0.90 vs 0.87) and on held-out mushrooms and trees (0.75 vs 0.67)."),
 "lat1_L1_s1": ("prop", "Base, seed 1", "works", "Replicate of the base: 0.88 vs 0.87, held-out 0.72 vs 0.67."),
 "lat1_L1_mem1": ("prop", "One memory channel", "works", "As good as eight channels."),
 "lat2_C0": ("prop", "No memory at all", "works", "Purely reactive bots do as well as bots with memory: memory is not needed here."),
 "lat1_L1_temp05": ("prop", "Sharper action choice (temperature 0.5)", "works", "Same level as the base."),
 "lat1_L1_T300": ("prop", "Longer build (300 steps)", "works", "Same level as the base; more time does not change the strategy."),
 "lat1_L1_spares100": ("prop", "Double spares", "works", "Best all-shape run (0.94 vs 0.87, held-out 0.88 vs 0.67). Colonnades: thin pillars with one-cell gaps under decks and caps."),
 "lat1_L1_spares0": ("prop", "No spares", "control", "Nothing to prop with: stays below the reference (0.85 vs 0.87). Shows that the gain elsewhere comes from spares used as props."),
 "lat1_L3": ("limit", "Cantilever limit 3", "control", "Little needs propping at L = 3: at the reference (0.96 vs 0.96)."),
 "lat1_L64_nolimit": ("limit", "No cantilever limit", "control", "Nothing needs propping: near-perfect builds, no props."),
 "lat2_L0": ("limit", "L = 0, all shapes (only columns hold)", "partial", "Builds staircases of spares to climb, but well below the reference (0.59 vs 0.80)."),
 "lat2_ovh_L0": ("limit", "L = 0, overhang shapes only", "fails", "Staircase ramps against every stem; overhangs are never propped (0.56 vs 0.64)."),
 "lat1_L1_overhangs_only": ("ovh", "Overhang shapes only (bridge, T, gamma)", "works", "Pillars under bridge decks and T arms: 0.80 vs 0.72."),
 "lat2_ovh_s1": ("ovh", "Overhang shapes only, seed 1", "works", "Replicate: 0.80 vs 0.72."),
 "lat2_ovh_s2": ("ovh", "Overhang shapes only, seed 2", "works", "Third seed: 0.82 vs 0.72."),
 "lat2_ovh_spares100": ("ovh", "Overhang shapes, double spares", "works", "Solid falsework blocks under every overhang: 0.88 vs 0.72; 16% of target cells stand only on props."),
 "lat1_L1_damage": ("robust", "Damage at half time (trained with it)", "partial", "Top half of the target cut away mid-build on half the episodes; rebuilds partly."),
 "lat2_kick10": ("robust", "Kick: 10% of bots removed after the build", "partial", "Learns, but ends below the reference (0.77 vs 0.87)."),
 "lat5_ovh_kick_warm": ("robust", "Kick on overhang shapes, warm start", "fails", "Restart of a collapsed run from the trained overhang controller: learns again but ends below the reference (0.67 vs 0.72)."),
 "lat2_switch": ("robust", "Target switches at half time", "fails", "Reconfiguration is hard: 0.73 at generation 900 vs 0.87, stuck."),
 "lat2_big": ("robust", "Bigger world (32 x 64)", "works", "Same rules, more room and more bots: 0.88, at the reference (0.87)."),
 "lat4_cure_noclean": ("cure", "Curing, no clean-site cost (from scratch)", "partial", "Bots that stay still 20 steps become rigid; learns without any cost on leftovers."),
 "lat4b_cure_clean_warm": ("cure", "Curing + clean-site cost, warm", "partial", "Clears most leftovers but loses some fill (0.77 vs 0.72)."),
 "lat4b_cure_clean_warm_s1": ("cure", "Curing + clean-site cost, warm, seed 1", "works", "Cure-and-extend: builds each overhang a cell at a time, letting each cure; final 0.92 with about 3 bots left on site. No build-then-strike."),
 "lat4b_nocure_clean_warm": ("cure", "Clean-site cost without curing (control)", "control", "Clearing the props drops the overhangs: final 0.69, below the reference (0.72)."),
 "lat5_formwork_clean_warm": ("cure", "Formwork curing + clean cost, warm", "fails", "A bot only cures while something is under it, as concrete needs formwork. Prop, cure, strike never appeared: it dropped the overhangs (0.72 = reference). Stopped."),
 "lat5_formwork_clean_warm_s1": ("cure", "Formwork curing + clean cost, warm, seed 1", "fails", "Replicate of the formwork test: same, 0.73 vs 0.72. Stopped."),
 "lat6_chasm_cure_warm": ("chasm", "Chasm, curing, warm start", "works", "Lays a cured deck across the gap from one anchor bot, walks over it, builds the far tower: 0.97 vs 0.50. With twice the time it crosses gaps up to 20 (trained on 3 to 12)."),
 "lat6_chasm_cure_warm_s1": ("chasm", "Chasm, curing, warm start, seed 1", "works", "Replicate: every deck complete, towers partly built at generation 100 (0.92)."),
 "lat6_chasm_nocure_warm": ("chasm", "Chasm, no curing, L = 1 (control)", "control", "Cannot cross: one deck cell, the rest of the swarm falls into the gap (0.06)."),
 "lat6_chasm_L3_nocure_warm": ("chasm", "Chasm, no curing, L = 3 (control)", "control", "About three deck cells, then nothing: 0.20. The far tower is never reached."),
 "lat3_height_L1": ("free", "Target-free: reach as high as possible", "works", "Trivial: a tower to the ceiling within 100 generations. Stopped."),
 "lat3_shelter_L1": ("free", "Target-free: shelter (cover empty cells), L = 1", "partial", "No roof on pillars: the swarm spreads into a low porous mat full of one-cell cavities (about 0.25 covered cells per bot)."),
 "lat3_shelter_L1_s1": ("free", "Target-free: shelter, L = 1, seed 1", "partial", "Same low porous mat."),
 "lat3_shelter_L3": ("free", "Target-free: shelter, L = 3", "partial", "Same low mat even though longer roofs would hold."),
 "lat3_shelter_L64": ("free", "Target-free: shelter, no limit", "running", ""),
 "lat2_movecost": ("collapse", "Move cost 0.02, from scratch", "fails", "Froze: no moves, fill 0. The cost is felt before any reward."),
 "lat5_movecost_warm": ("collapse", "Move cost 0.005, warm start", "works", "Restart from the base controller: keeps 0.90 fill, moves barely change."),
 "lat2_ovh_kick10": ("collapse", "Kick on overhang shapes, from scratch", "fails", "Stuck at fill 0.13."),
 "lat4_cure_clean": ("collapse", "Curing + clean cost, from scratch", "fails", "Learned to stay out of the build site: fill 0."),
 "lat4_cure_clean_s1": ("collapse", "Curing + clean cost, from scratch, seed 1", "fails", "Same: stays home, fill 0."),
 "lat4_nocure_clean": ("collapse", "Clean cost, no curing, from scratch", "fails", "Same local optimum."),
 "lat1_L1_pop256": ("knobs", "ES population 256", "inconclusive", "Stopped at generation 110 to free the GPU; nothing distinctive."),
 "lat1_L1_sigma02": ("knobs", "ES noise 0.02", "inconclusive", "Stopped at generation 200."),
 "lat1_L1_wide64": ("knobs", "Network width 64", "inconclusive", "Stopped at generation 175."),
}
GROUPS = [
 ("ref", "References", "What doing nothing useful looks like, and what the target alone keeps under gravity."),
 ("prop", "Does propping emerge? All shapes, cantilever limit 1", "Training objective is target fill only. Fill above the perfect-placement reference means bots outside the target are holding it up."),
 ("ovh", "Overhang shapes only", "Trained on bridges, T and gamma shapes, where most cells need support."),
 ("limit", "Changing the cantilever limit", "L = number of cells a bot may stick out sideways and stay held."),
 ("cure", "Curing and falsework removal", "Bots that stay still become rigid; a cost on bots left in the build site asks the swarm to clean up."),
 ("chasm", "Chasm crossing", "A gap in the floor: a deck across it and a tower on the far side. Bots that fall in are lost. The 0.50 reference (target placed, then settled) cannot be reached without crossing."),
 ("robust", "Robustness, reconfiguration, scale", ""),
 ("free", "Target-free objectives", "No target shape: the swarm is scored on height or on how much empty space it covers."),
 ("collapse", "Collapsed runs: cost felt before reward", "Each learned to do nothing. Restarted from trained controllers where it mattered."),
 ("knobs", "Optimizer knobs (stopped early)", ""),
]
EARLY = [
 ("s5_ext_fb16", "Continuous physics, drone forces", "partial", "Shape formation works when bots are pushed by external forces (build IoU 0.61), but structure does not stand on its own."),
 ("s6_g1_fb2_hold3", "Continuous physics, internal forces under gravity", "fails", "Never leaves the pile; most bots fall."),
 ("s7_g1_fb2_T128_nograv", "Continuous physics, no gravity (control)", "control", "Indistinguishable from the gravity-trained runs: gravity was not shaping anything."),
]


def chips(cfg):
    out = []
    for k, b in BASE.items():
        v = cfg.get(k, b)
        if v != b and not (k == "families" and v is None):
            if k == "families":
                v = ", ".join(v)
            elif k == "init_from":
                v = os.path.basename(v).replace(".pt", "")
            elif k == "obj":
                v = v
            out.append(f'<span class="chip"><span class="k">{LAB[k]}</span> {html.escape(str(v))}</span>')
    return "".join(out) or '<span class="chip base">base settings</span>'


def metrics(info):
    m = info.get("metrics", {})
    cfg = info["cfg"]
    rows = []
    fin = m.get("final", {})
    if fin.get("val"):
        v = fin["val"]; src = "final"
    elif m.get("last_eval"):
        v = m["last_eval"]; src = f"gen {m['last_eval_gen']}"
    else:
        v = None; src = ""
    if v and cfg.get("obj", "fill") == "fill":
        rows.append(("val fill", f"{v['fill']:.2f}", f"ref {v['ref_target_stands']:.2f}", v['fill'] > v['ref_target_stands'] + 0.005))
        if "propped" in v:
            rows.append(("propped", f"{v['propped']:.2f}", "of target", None))
        if cfg.get("clean_w", 0) > 0 and "site_out" in v:
            rows.append(("left in site", f"{v['site_out']:.0f}", "bots", None))
    elif v:
        rows.append((cfg["obj"], f"{v.get(cfg['obj'], 0):.2f}", "", None))
    h = fin.get("held")
    if h and cfg.get("obj", "fill") == "fill":
        rows.append(("held-out fill", f"{h['fill']:.2f}", f"ref {h['ref_target_stands']:.2f}", h['fill'] > h['ref_target_stands'] + 0.005))
    sf = info.get("show_fill", {})
    if sf and cfg.get("obj", "fill") == "fill":
        rows.append(("these 6 shapes", f"{sum(sf.values()) / len(sf):.2f}", "mean fill", None))
    cells = []
    for name, val, sub, good in rows:
        cls = "" if good is None else (" above" if good else " below")
        cells.append(f'<div class="m{cls}"><div class="mv">{val}</div><div class="ml">{name}<br><span>{sub}</span></div></div>')
    gen = m.get("gen", 0)
    st = info["status"]
    prog = "" if st == "reference" else f'{st} · gen {gen + (1 if st == "done" else 0)}/{cfg["gens"]}' + (f" · metrics at {src}" if src else "")
    return "".join(cells), prog


def card(name, info):
    grp, title, verdict, note = V.get(name, ("other", name, "inconclusive", ""))
    if info["status"] == "running" and verdict not in ("running",):
        vlabel = verdict + " so far"
    else:
        vlabel = verdict
    mcells, prog = metrics(info)
    return f'''<article class="card" data-verdict="{verdict}" id="{name}">
  <img src="gifs/{name}.gif" alt="{html.escape(title)}: six shapes being built" loading="lazy" width="624" height="276">
  <div class="body">
    <div class="top"><h3>{html.escape(title)}</h3><span class="pill {verdict}">{vlabel}</span></div>
    <div class="id"><code>{name}</code> <span class="prog">{prog}</span></div>
    <div class="chips">{chips(info["cfg"]) if info["status"] != "reference" else ""}</div>
    <div class="metrics">{mcells}</div>
    <p class="note">{html.escape(note)}</p>
  </div>
</article>'''


def main():
    infos = {}
    for f in sorted(os.listdir(G)):
        if f.endswith(".json"):
            i = json.load(open(os.path.join(G, f)))
            if os.path.exists(os.path.join(G, i["name"] + ".gif")):
                infos[i["name"]] = i
    # summary table
    trs = []
    for gk, gt, _ in GROUPS:
        for name, info in infos.items():
            g, title, verdict, note = V.get(name, ("other", name, "inconclusive", ""))
            if g != gk or gk == "ref":
                continue
            m = info.get("metrics", {})
            v = m.get("final", {}).get("val") or m.get("last_eval") or {}
            h = m.get("final", {}).get("held") or {}
            fill = f"{v['fill']:.2f} / {v['ref_target_stands']:.2f}" if "fill" in v and info["cfg"]["obj"] == "fill" else (f"{info['cfg']['obj']} {v.get(info['cfg']['obj'], 0):.2f}" if v else "–")
            held = f"{h['fill']:.2f} / {h['ref_target_stands']:.2f}" if h and info["cfg"]["obj"] == "fill" else "–"
            trs.append(f'<tr data-verdict="{verdict}"><td><a href="#{name}">{html.escape(title)}</a></td><td><span class="pill {verdict}">{verdict}</span></td>'
                       f'<td class="num">{fill}</td><td class="num">{held}</td><td>{info["status"]}</td></tr>')
    sections = []
    for gk, gt, gd in GROUPS:
        cards = [card(n, i) for n, i in infos.items() if V.get(n, ("other",))[0] == gk]
        if cards:
            sections.append(f'<section><h2>{html.escape(gt)}</h2>' + (f'<p class="lede">{html.escape(gd)}</p>' if gd else "") +
                            f'<div class="grid">{"".join(cards)}</div></section>')
    early = "".join(f'''<article class="card" data-verdict="{v}"><img src="early/{n}.gif" alt="{html.escape(t)}" loading="lazy">
  <div class="body"><div class="top"><h3>{html.escape(t)}</h3><span class="pill {v}">{v}</span></div><div class="id"><code>{n}</code></div>
  <p class="note">{html.escape(note)}</p></div></article>''' for n, t, v, note in EARLY if os.path.exists(os.path.join(G, "early", n + ".gif")))
    if early:
        sections.append('<section><h2>Earlier track: continuous disc physics (paused)</h2><p class="lede">Different simulator and colours (see each GIF). Kept here for comparison: this is what did not work before the lattice.</p>'
                        f'<div class="grid">{early}</div></section>')
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery_template.html")).read()
    page = page.replace("{{TABLE}}", "\n".join(trs)).replace("{{SECTIONS}}", "\n".join(sections)).replace("{{N}}", str(len(infos)))
    open(os.path.join(G, "index.html"), "w").write(page)
    print("wrote", os.path.join(G, "index.html"), len(infos), "runs")


if __name__ == "__main__":
    main()
