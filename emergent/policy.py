"""The controller copied into every bot: a radius-limited message-passing network with a recurrent memory.

Per bot it sees: the target map at its own position (6 channels), its velocity, its height above the floor (capped),
its memory h; and, from every neighbour within r_sense: relative position, distance, relative velocity, the
neighbour's embedding (which includes the neighbour's memory). It outputs a force request (2, bounded by f_act per
axis), a grip in [0, 1] and a new memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class Policy(nn.Module):
    def __init__(self, H=32, M=96, rounds=2, r_sense=3.0, map_ch=6, f_act=3.0, use_h=True, see_v=True):
        super().__init__()
        self.H, self.M, self.rounds, self.r_sense, self.f_act, self.use_h, self.see_v = H, M, rounds, r_sense, f_act, use_h, see_v
        self.enc = nn.Sequential(nn.Linear(map_ch + 2 + 2 + H, M), nn.SiLU(), nn.Linear(M, M))
        self.ea = nn.ModuleList([nn.Linear(M, M) for _ in range(rounds)])          # receiver
        self.eb = nn.ModuleList([nn.Linear(M, M, bias=False) for _ in range(rounds)])  # sender
        self.eg = nn.ModuleList([nn.Linear(6, M, bias=False) for _ in range(rounds)])  # geometry
        self.e2 = nn.ModuleList([nn.Linear(M, M) for _ in range(rounds)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * M, M), nn.SiLU(), nn.Linear(M, M)) for _ in range(rounds)])
        self.norm = nn.ModuleList([nn.LayerNorm(M) for _ in range(rounds)])
        self.gru = nn.GRUCell(M, H)
        self.head = nn.Sequential(nn.Linear(M + H, M), nn.SiLU(), nn.Linear(M, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, v, h, mapf, alive):
        """x, v: (B, N, 2); h: (B, N, H); mapf: (B, N, 6) map features at x; alive: (B, N)."""
        B, N, _ = x.shape
        vin = v if self.see_v else torch.zeros_like(v)
        y = x[..., 1:2]
        own = torch.cat([mapf, vin, torch.clamp(y, max=3.0) / 3.0, (y < 0.65).float(), h if self.use_h else torch.zeros_like(h)], -1)
        e = self.enc(own)
        d = x[:, None, :, :] - x[:, :, None, :]                 # [b, i, j] = x_j - x_i
        dist = torch.sqrt((d * d).sum(-1) + 1e-8)
        w = torch.clamp((self.r_sense - dist) / 0.5, 0, 1) * alive[:, :, None] * alive[:, None, :]
        w = w * (1 - torch.eye(N, device=x.device))
        dv = vin[:, None, :, :] - vin[:, :, None, :]
        geo = torch.cat([d / self.r_sense, dist[..., None] / self.r_sense, dv, (dist[..., None] < 1.15).float()], -1)
        cnt = w.sum(2, keepdim=True)
        for r in range(self.rounds):
            z = Fn.silu(self.ea[r](e)[:, :, None, :] + self.eb[r](e)[:, None, :, :] + self.eg[r](geo))
            m = self.e2[r]((w[..., None] * z).sum(2) / (1 + cnt))
            e = self.norm[r](e + self.upd[r](torch.cat([e, m], -1)))
        hn = self.gru(e.reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1) if self.use_h else h
        o = self.head(torch.cat([e, hn], -1))
        a = self.f_act * torch.tanh(o[..., :2])
        grip = torch.sigmoid(o[..., 2] + 2.0)       # starts gripping (about 0.88)
        return a, grip, hn


def map_features(maps, x, W):
    """maps: (B, C, R, R), row index = y. x: (B, N, 2) world coords -> (B, N, C) bilinear samples."""
    g = (x / W) * 2 - 1
    s = Fn.grid_sample(maps.float(), g[:, :, None, :], mode="bilinear", padding_mode="border", align_corners=False)
    return s[..., 0].transpose(1, 2)


def build_maps(sdf, occ, W):
    """sdf, occ: (B, R, R) tensors -> (B, 6, R, R): sdf/4 (clipped), sdf gradient (2), occupancy, two blurs."""
    R = sdf.shape[-1]
    cell = W / R
    s = sdf.float()[:, None]
    gy, gx = torch.gradient(s[:, 0], spacing=cell, dim=(1, 2))
    o = occ.float()[:, None]

    def blur(img, sigma_world):
        sg = sigma_world / cell
        k = int(3 * sg) * 2 + 1
        t = torch.arange(k, device=img.device, dtype=torch.float32) - k // 2
        g1 = torch.exp(-0.5 * (t / sg) ** 2)
        g1 = g1 / g1.sum()
        img = Fn.conv2d(Fn.pad(img, (k // 2, k // 2, 0, 0), mode="replicate"), g1.view(1, 1, 1, k))
        return Fn.conv2d(Fn.pad(img, (0, 0, k // 2, k // 2), mode="replicate"), g1.view(1, 1, k, 1))

    return torch.cat([torch.clamp(s / 4.0, -2, 2), gx[:, None], gy[:, None], o, blur(o, 1.5), blur(o, 4.0)], 1)
