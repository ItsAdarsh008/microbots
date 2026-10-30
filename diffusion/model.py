"""Set denoiser for particles: velocity field v(x_i, t | other particles, map).

One model covers the whole ladder of how much a particle may know:
  radius=None : every particle attends to every other (global communication; upper bound)
  radius=r    : a particle attends only to particles within distance r (swarm-realisable);
                `layers` is then the number of message rounds per physical step
  cond='map'  : each particle reads the target map at its own position (sdf, its gradient,
                occupancy at three blur scales), like a Kilobot that carries the shape map
  time=False  : the network is not told the denoising time (a swarm has no shared clock)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

R = 64


def make_maps(occ, sdf, coarse=0):
    """occ, sdf: (B, R, R) -> (B, 6, R, R): sdf, d sdf/dx, d sdf/dy, occ, occ blurred at 2 and 4 px.
    coarse=g>0: the swarm is only told a g x g occupancy grid (g*g numbers, smoothly upsampled); the
    signed distance is replaced by a smooth proxy of that coarse occupancy."""
    if coarse:
        occ = F.interpolate(F.adaptive_avg_pool2d(occ[:, None], coarse), size=(R, R), mode="bilinear",
                            align_corners=False)[:, 0]
        sdf = (0.5 - occ) * 0.5
    px = 2.0 / R
    gy, gx = torch.gradient(sdf, spacing=px, dim=(1, 2))

    def blur(x, s):
        k = int(3 * s) * 2 + 1
        g = torch.exp(-0.5 * ((torch.arange(k, device=x.device) - k // 2) / s) ** 2)
        g = (g / g.sum()).to(x.dtype)
        x = x[:, None]
        x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
        x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
        return x[:, 0]
    return torch.stack([sdf, gx, gy, occ, blur(occ, 2.0), blur(occ, 4.0)], 1)


def read_map(pos, maps):
    """Sample the maps at particle positions. pos (B,N,2) world coords in [-1,1] (may leave the box)."""
    c = pos.clamp(-1, 1)
    out = F.grid_sample(maps, c[:, :, None, :], mode="bilinear", padding_mode="border",
                        align_corners=False)[..., 0].transpose(1, 2)        # (B, N, 6)
    outside = (pos - c).norm(dim=-1)
    return torch.cat([out[..., :1] + outside[..., None], out[..., 1:]], -1)  # sdf grows outside the box


def fourier(x, n=8):
    f = (2.0 ** torch.arange(n, device=x.device)) * math.pi / 2
    a = x[..., None] * f
    return torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.n1 = nn.LayerNorm(d, elementwise_affine=False)
        s.n2 = nn.LayerNorm(d, elementwise_affine=False)
        s.qkv = nn.Linear(d, 3 * d)
        s.o = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.ada = nn.Linear(d, 4 * d)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)

    def forward(s, x, temb, bias, mask):
        B, N, D = x.shape
        m = s.ada(temb)
        a1, b1, a2, b2 = (m[:, None] if m.dim() == 2 else m).chunk(4, -1)
        y = s.n1(x) * (1 + a1) + b1
        q, k, v = s.qkv(y).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // s.h) + bias
        if mask is not None:
            att = att.masked_fill(~mask[:, None], float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + s.o(y)
        y = s.n2(x) * (1 + a2) + b2
        return x + s.mlp(y)



GRID_LO, GRID_HI = -1.6, 1.6   # the shared grid covers the arena the particles can wander in


def splat(x, vals, g):
    """Bilinear scatter of per-particle values (B,N,C) onto a g x g grid -> sums (B,C,g,g), weights (B,1,g,g)."""
    B, N, C = vals.shape
    p = ((x.clamp(GRID_LO, GRID_HI) - GRID_LO) / (GRID_HI - GRID_LO) * (g - 1)).float()
    i0 = p.floor().clamp(0, g - 2); f = p - i0; i0 = i0.long()
    out = torch.zeros(B, C + 1, g * g, device=x.device, dtype=torch.float32)
    v = torch.cat([vals.float(), torch.ones(B, N, 1, device=x.device)], -1).transpose(1, 2)     # (B, C+1, N)
    for dy in (0, 1):
        for dxx in (0, 1):
            w = (f[..., 0] if dxx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])    # (B,N)
            idx = (i0[..., 1] + dy) * g + (i0[..., 0] + dxx)
            out.scatter_add_(2, idx[:, None].expand(-1, C + 1, -1), v * w[:, None])
    out = out.view(B, C + 1, g, g)
    return out[:, :C], out[:, C:]


def read_grid(x, field):
    """Bilinear read of a (B,C,g,g) field at particle positions -> (B,N,C)."""
    c = (x.clamp(GRID_LO, GRID_HI) - GRID_LO) / (GRID_HI - GRID_LO) * 2 - 1
    return F.grid_sample(field, c[:, :, None, :].float(), mode="bilinear", padding_mode="border",
                         align_corners=True)[..., 0].transpose(1, 2)


def density_feats(x, g, swap):
    """Swarm density histogram on a g x g grid (counts, uniform swarm = 1), read at own position with its gradient.
    This is a raw mean-field state: g*g numbers, no learned parameters."""
    _, w = splat(x, x[..., :0], g)
    dm = w * (g * g / x.shape[1])
    if swap:
        dm = dm.roll(1, 0)
    gy, gx = torch.gradient(dm[:, 0], dim=(1, 2))
    return read_grid(x, torch.cat([dm, gx[:, None], gy[:, None]], 1))


class GlobalMix(nn.Module):
    """One round of shared state: every node contributes, every node reads. Bandwidth in floats per round:
    mean: gdim | isab: gtok*gdim | grid: g*g*gc.  `swap` replaces the shared state by that of another swarm in
    the batch (a control: same parameters, but the broadcast carries no information about THIS swarm)."""
    def __init__(s, d, kind, gdim, gtok, g, gc):
        super().__init__()
        s.kind, s.g = kind, g
        if kind == "mean":
            s.down = nn.Linear(d, gdim); s.up = nn.Sequential(nn.GELU(), nn.Linear(gdim, d)); last = s.up[1]
        elif kind == "isab":
            s.q = nn.Parameter(torch.randn(gtok, d) * 0.02); s.kv = nn.Linear(d, 2 * d)
            s.tdown = nn.Linear(d, gdim); s.tup = nn.Linear(gdim, d); s.nq = nn.Linear(d, d); s.out = nn.Linear(d, d); last = s.out
        elif kind == "grid":
            s.proj = nn.Linear(d, gc)
            s.net = nn.Sequential(nn.Conv2d(gc, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(16, 16, 3, padding=1))
            s.up = nn.Linear(16, d); last = s.up
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(s, h, x, swap):
        B, N, D = h.shape
        if s.kind == "mean":
            p = s.down(h.mean(1))
            if swap:
                p = p.roll(1, 0)
            return s.up(p)[:, None]
        if s.kind == "isab":
            k, v = s.kv(h).chunk(2, -1)
            a = (s.q[None].expand(B, -1, -1) @ k.transpose(1, 2) / math.sqrt(D)).softmax(-1) @ v      # (B,m,D)
            t = s.tdown(a)
            if swap:
                t = t.roll(1, 0)
            tk = s.tup(t)
            b = (s.nq(h) @ tk.transpose(1, 2) / math.sqrt(D)).softmax(-1) @ tk
            return s.out(b)
        sums, w = splat(x, s.proj(h), s.g)
        field = sums / (w + 1e-2)
        if swap:
            field = field.roll(1, 0)
        return s.up(read_grid(x, s.net(field))).to(h.dtype)


class Denoiser(nn.Module):
    def __init__(s, d=192, heads=6, layers=6, radius=None, cond="none", time=True,
                 gstate="none", gdim=16, gtok=4, ggrid=8, gc=1, grounds=None):
        super().__init__()
        s.grounds = layers if grounds is None else grounds     # shared state is used in the last `grounds` layers only
        s.radius, s.cond, s.time, s.L, s.H = radius, cond, time, layers, heads
        s.gkind, s.ggrid, s.gswap = gstate, ggrid, False
        s.inp = nn.Sequential(nn.Linear(34 + (6 if cond == "map" else 0) + (3 if gstate == "dens" else 0), d), nn.GELU(), nn.Linear(d, d))
        s.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        s.rel = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, layers * heads))
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)
        s.gmix = nn.ModuleList([GlobalMix(d, gstate, gdim, gtok, ggrid, gc) for _ in range(layers)]) if gstate in ("mean", "isab", "grid") else None

    def forward(s, x, t, maps=None):
        B, N, _ = x.shape
        feats = [fourier(x, 8)]
        if s.cond == "map":
            feats.append(read_map(x, maps))
        if s.gkind == "dens":
            feats.append(density_feats(x, s.ggrid, s.gswap).to(x.dtype))
        h = s.inp(torch.cat(feats, -1))
        if s.time:
            fr = torch.exp(-math.log(10000) * torch.arange(32, device=x.device) / 32)
            a = t[..., None] * 1000 * fr
            te = s.temb(torch.cat([a.sin(), a.cos()], -1))
        else:
            te = torch.zeros(B, h.shape[-1], device=x.device)
        dx = x[:, None, :, :] - x[:, :, None, :]                       # dx[b,i,j] = x_j - x_i
        dist = dx.norm(dim=-1, keepdim=True)
        scale = s.radius if s.radius else 1.0
        bias_all = s.rel(torch.cat([dx / scale, dist / scale], -1)).view(B, N, N, s.L, s.H)
        mask = (dist[..., 0] < s.radius) if s.radius else None
        for l, blk in enumerate(s.blocks):
            h = blk(h, te, bias_all[:, :, :, l].permute(0, 3, 1, 2), mask)
            if s.gmix is not None and l >= s.L - s.grounds:
                h = h + s.gmix[l](h, x, s.gswap)
        return s.out(h)


def build(cfg):
    """Construct the denoiser from a run config (every script uses this)."""
    m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"],
                 gstate=cfg.get("gstate", "none"), gdim=cfg.get("gdim", 16), gtok=cfg.get("gtok", 4),
                 ggrid=cfg.get("ggrid", 8), gc=cfg.get("gc", 1), grounds=cfg.get("grounds"))
    m.gswap = bool(cfg.get("gshuffle", False))
    return m
