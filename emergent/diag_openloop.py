"""Diagnostic: optimise free per-bot force sequences (no controller) through the physics to form shapes.
If this cannot move a pile into a shape, the problem is the physics/gradients, not the network."""
import sys, time, torch
import train
from sim import Physics, pile
kw = dict(a.split("=") for a in sys.argv[1:])
decay = float(kw.pop("decay", 1.0))
iters = int(kw.pop("iters", 301))
phys_kw = {k: (v if k == "reaction" else float(v)) for k, v in kw.items()}
dev = "cuda"
cfg = dict(train.DEF); cfg.update(grad_decay=decay, grad_clip_step=10.0)
data = train.Data("data/world128.npz", dev)
S = data.sets["val"]; idx = torch.arange(0, 192, 24)[:8]
maps, Q, occ = S["maps"][idx].float(), S["pts"][idx], S["occ"][idx]
B, N = len(idx), data.N
ph = Physics(W=data.W, **phys_kw)
print(phys_kw, flush=True)
ph.step = torch.compile(ph.step, dynamic=False)
x0 = pile(B, N, W=data.W).to(dev)
T, K = 64, 4
u = torch.zeros(B, N, T, 2, device=dev, requires_grad=True)
gu = torch.full((B, N, T), -2.0, device=dev, requires_grad=True)
g = torch.full((B, N), 1.0, device=dev); al = torch.ones(B, N, device=dev)
opt = torch.optim.Adam([u, gu], lr=0.1)
for it in range(iters):
    x, v = x0, torch.zeros_like(x0)
    for t in range(T):
        a = 3.0 * torch.tanh(u[:, :, t])
        g = torch.sigmoid(gu[:, :, t])
        for _ in range(K):
            x, v = ph.step(x, v, a, g, al)
        x, v = train.tame(x, cfg), train.tame(v, cfg)
    L, parts = train.shape_loss(x, al, Q, maps, data.W, 1.0)
    opt.zero_grad(); L.backward()
    gn = u.grad.norm().item()
    torch.nn.utils.clip_grad_norm_([u, gu], 1.0); opt.step()
    if it % 25 == 0:
        with torch.no_grad():
            fp = train.footprint(x, al, data.R, data.W)
            print(f"it {it} loss {L.item():.2f} {parts} gn {gn:.1f} IoU {train.iou(fp, occ).mean().item():.3f} "
                  f"grip {g.mean().item():.2f} mean disp {(x - x0).norm(dim=-1).mean().item():.2f}", flush=True)
