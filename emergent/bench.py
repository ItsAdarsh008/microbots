"""Timing of one training iteration's pieces (for speed work)."""
import time, torch, sys
from sim import Physics, pile
from policy import Policy, map_features
import train
cfg = dict(train.DEF); cfg["name"] = "bench"
dev = "cuda"
data = train.Data("data/tiny.npz", dev)
ph = Physics(W=data.W)
pol = Policy().to(dev)
if "--compile" in sys.argv:
    ph.step = torch.compile(ph.step, dynamic=False)
    pol = torch.compile(pol, dynamic=False)
S = data.sets["train"]; maps, Q = S["maps"][:16].float(), S["pts"][:16]
x0 = pile(16, data.N, W=data.W).to(dev)
def T(f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n
def fwd():
    with torch.no_grad(): train.rollout(pol, ph, maps, x0, cfg, grad=False)
def full():
    r = train.rollout(pol, ph, maps, x0, cfg, grad=True)
    l, _ = train.shape_loss(r["xh"], r["alive"], Q, maps, data.W, 1.0); l.backward()
x = x0.clone(); v = torch.zeros_like(x); g = torch.ones(16, data.N, device=dev); a = torch.zeros_like(x)
def phys():
    for _ in range(100): ph.step(x, v, a, g, g)
h = torch.zeros(16, data.N, 32, device=dev)
def polf():
    with torch.no_grad():
        for _ in range(10): pol(x, v, h, map_features(maps, x, data.W), g)
print("physics step ms", T(phys) * 10)
print("policy call ms", T(polf) * 100)
print("rollout no grad s", T(fwd))
print("rollout + backward s", T(full))
