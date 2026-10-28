"""Put the trained weights from policy.npz into the browser sandbox.

    python linking/learn/export_policy.py      # updates linking/sandbox/index.html

The weights go between the POLICY-START and POLICY-END markers in the page,
packed as base64 float32, layer by layer (w0, b0, w1, b1, ...).

The weights are packed as base64 float32, layer by layer (w0, b0, w1, b1, ...).
"""
import base64
import os

import numpy as np

HERE = os.path.dirname(__file__)
z = np.load(os.path.join(HERE, "policy.npz"))
n = len([k for k in z.files if k.startswith("w")])
sizes = [int(z["w0"].shape[0])] + [int(z[f"w{i}"].shape[1]) for i in range(n)]
flat = np.concatenate([np.concatenate([z[f"w{i}"].ravel(), z[f"b{i}"].ravel()]) for i in range(n)]).astype("<f4")
data = base64.b64encode(flat.tobytes()).decode()
page = os.path.join(HERE, "..", "sandbox", "index.html")
html = open(page).read()
a, b = html.index("// POLICY-START"), html.index("// POLICY-END")
block = ("// POLICY-START (written by linking/learn/export_policy.py from policy.npz)\n"
         f"const POLICY_SIZES = {sizes};\n"
         f'const POLICY_DATA = "{data}";\n')
open(page, "w").write(html[:a] + block + html[b:])
print("updated", page, f"({flat.size} weights, sizes {sizes})")
