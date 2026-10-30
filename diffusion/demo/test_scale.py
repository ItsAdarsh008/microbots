"""Does a swarm trained with 256 bots work with more?  More bots, sensing range kept fixed in bot spacings (server on localhost:8000)."""
import json, time, urllib.request
def post(d):
    r = urllib.request.Request("http://localhost:8000/api/run", data=json.dumps(d).encode(), method="POST")
    return json.load(urllib.request.urlopen(r, timeout=300))
print(f"{'model':22s} {'shape':12s}" + "".join(f"{('N=' + str(n)):>30s}" for n in (256, 512, 1024, 2048)))
for model, start in [("s14_refv3_r25_g16", "pile_c"), ("s6_r60_aux_in100_rep100", "noise")]:
    for shape in ["star", "anchor", "snowflake", "letter K (thin)", "letter B (thick)"]:
        row = f"{model[:22]:22s} {shape:12s}"
        for n in (256, 512, 1024, 2048):
            t = time.time(); j = post(dict(model=model, shape=shape, start=start, k=16, n=n, keep_spacings=True))
            row += f"   in {j['inside']:.2f} cov {j['cover']:.2f} coll {j['coll']:.2f} {time.time()-t:3.1f}s"
        print(row, flush=True)
