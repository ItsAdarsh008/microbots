"""Batched, differentiable 2D physics for disc bots in a vertical plane (torch, dense pairs).

Units: bot diameter 1, mass 1, gravity g (default 1 = one bot weight per unit force).
Forces on each bot:
  gravity; contact (penalty spring + normal damping) with neighbours, floor (y = 0) and side walls;
  smooth Coulomb friction (bot-bot and bot-floor);
  bonds: if two bots both grip (grip in [0, 1]), they pull together with a tension that saturates at
         f_bond * min(grip_i, grip_j) and fades to zero between r_hold and r_bond (the bond breaks) -- a latch of
         limited strength, central force only (no bending stiffness anywhere);
  actuation: each bot asks for a force a_i (|a_i| <= f_act per axis). In 'internal' mode it can only push against
         what it touches: the applied force is scaled by min(1, touch weight) and the reaction goes to the touched
         neighbours (the floor's share is absorbed by the ground), so momentum is conserved. 'external' mode is an
         unphysical control (drones).
Dead bots (alive = 0) feel nothing and exert nothing.
"""
import torch


class Physics:
    def __init__(self, g=1.0, k_c=200.0, c_c=3.0, mu=0.5, f_bond=4.0, r_hold=1.15, r_bond=1.25, c_b=2.0,
                 f_act=3.0, reaction="internal", r_touch=1.15, air=0.05, dt=0.02, W=24.0, v_slip=0.05):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def pairs(self, x, alive):
        d = x[:, None, :, :] - x[:, :, None, :]                 # d[b, i, j] = x_j - x_i
        dist = torch.sqrt((d * d).sum(-1) + 1e-8)
        N = x.shape[1]
        pm = alive[:, :, None] * alive[:, None, :] * (1 - torch.eye(N, device=x.device))
        return d, dist, pm

    def bond_weight(self, dist, grip, pm):
        """Smooth bond activity in [0, 1] per pair: min grip times the distance window."""
        win = torch.clamp((self.r_bond - dist) / (self.r_bond - self.r_hold), 0, 1)
        return torch.minimum(grip[:, :, None], grip[:, None, :]) * win * pm

    def step(self, x, v, a, grip, alive, g_scale=1.0):
        d, dist, pm = self.pairs(x, alive)
        n = d / dist[..., None]
        vrel = v[:, None, :, :] - v[:, :, None, :]                # v_j - v_i
        vn = (vrel * n).sum(-1)
        vt = vrel - vn[..., None] * n
        # contact
        ov = torch.relu(1.0 - dist) * pm
        ind = (ov > 0).float()
        fn = self.k_c * ov - self.c_c * vn * ind                  # >0 pushes i away from j
        fn = torch.relu(fn)                                       # contacts do not pull
        F = -(fn[..., None] * n).sum(2)
        # bot-bot friction (smooth Coulomb on the tangential relative velocity)
        vtn = torch.sqrt((vt * vt).sum(-1) + 1e-8)
        F = F + (self.mu * fn * torch.tanh(vtn / self.v_slip) / vtn)[..., None].mul(vt).sum(2)
        # bonds: tension only, saturating, central
        wb = self.bond_weight(dist, grip, pm)
        stretch = torch.relu(dist - 1.0)
        cap = self.f_bond * wb
        T = cap * torch.tanh(200.0 * stretch / (self.f_bond + 1e-6)) + self.c_b * wb * vn
        T = torch.relu(T)
        F = F + (T[..., None] * n).sum(2)
        # floor
        y = x[..., 1]
        pen = torch.relu(0.5 - y) * alive
        Nf = torch.relu(self.k_c * pen - self.c_c * v[..., 1] * (pen > 0).float())
        Ff = torch.stack([-self.mu * Nf * torch.tanh(v[..., 0] / self.v_slip), Nf], -1)
        F = F + Ff
        # side walls
        xl = torch.relu(0.5 - x[..., 0]) - torch.relu(x[..., 0] - (self.W - 0.5))
        F = F + torch.stack([self.k_c * xl, torch.zeros_like(xl)], -1) * alive[..., None]
        # actuation
        if a is not None:
            if self.reaction == "internal":
                wt = torch.clamp((self.r_touch - dist) / (self.r_touch - 1.0), 0, 1) * pm
                wf = torch.clamp((0.65 - y) / 0.15, 0, 1) * alive
                Wsum = wt.sum(2) + wf
                ae = a * torch.clamp(Wsum, max=1.0)[..., None] * alive[..., None]
                share = wt / torch.clamp(Wsum, min=1e-6)[..., None]          # [b, i, j]: i's reaction share on j
                F = F + ae - torch.einsum("bij,bid->bjd", share, ae)
            else:
                F = F + a * alive[..., None]
        F = F - self.air * v
        F = F + torch.stack([torch.zeros_like(y), -self.g * g_scale * torch.ones_like(y)], -1)
        F = F * alive[..., None]
        v = v + self.dt * F
        x = x + self.dt * v
        return x, v

    def bonds(self, x, grip, alive, thr=0.5):
        """Bond graph used for analysis: active bonds with weight > thr."""
        _, dist, pm = self.pairs(x, alive)
        return self.bond_weight(dist, grip, pm) > thr


def pile(B, N, W=24.0, width=None, jitter=0.05, offset=0.0, gen=None, device="cpu"):
    """Bots stacked in a hexagonal block on the floor, centred at W/2 (+ offset)."""
    width = width or int(round((N * 1.6) ** 0.5))
    rows = (N + width - 1) // width
    ii = torch.arange(N, device=device)
    r, c = ii // width, ii % width
    xs = c.float() * 1.02 + 0.51 * (r % 2).float()
    ys = 0.5 + r.float() * 0.885
    xs = xs - xs.mean() + W / 2 + offset
    p = torch.stack([xs, ys], -1)[None].repeat(B, 1, 1)
    p = p + jitter * torch.randn(p.shape, generator=gen, device=device)
    return p
