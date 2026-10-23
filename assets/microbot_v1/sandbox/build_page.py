"""Inline sim.js and the CAD STLs into index.src.html -> index.html (the published single page)."""
import base64
import json
from pathlib import Path

here = Path(__file__).resolve().parent
sim = (here / "sim.js").read_text().replace("export function createSim", "function createSim")
stl = {n: base64.b64encode((here.parent / "stl" / f"part_{n}.stl").read_bytes()).decode()
       for n in ("body", "turret", "arm", "tip")}
page = (here / "index.src.html").read_text().replace("/*__SIM__*/", sim).replace("/*__STL__*/", json.dumps(stl))
(here / "index.html").write_text(page)
print("wrote index.html", len(page), "bytes")
