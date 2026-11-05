from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_sum(v: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, v.shape[-1], dtype=v.dtype, device=v.device)
    return out.index_add_(0, idx, v)


class MGHAMLayer(nn.Module):

    def __init__(self, d_in: int, d: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.inp = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d)                 
        self.gate = nn.Linear(d, d)               
        self.norm = nn.LayerNorm(d)               
        self.drop = nn.Dropout(dropout)

    def forward(self, x_prev: torch.Tensor, hg: dict):
        x_in = self.inp(x_prev)
        node, edge, m = hg["node"], hg["edge"], hg["n_edges"]
        de_inv, n = hg["de_inv"], x_in.shape[0]

        e = seg_sum(x_in[node], edge, m) * de_inv.unsqueeze(-1)

        heads = []
        for k in range(self.heads):
            sl = slice(k * self.dh, (k + 1) * self.dh)
            q = self.wq(e)[:, sl]                                   
            kk = self.wk(e)[:, sl]
            v = self.wv(e)[:, sl]
            a = torch.softmax(q @ kk.t() / (self.dh ** 0.5), dim=1)  
            e_m = a @ v                                              
            heads.append(seg_sum(e_m[edge], node, n))                
        x_m = self.wo(torch.cat(heads, -1))                          

        g = torch.sigmoid(self.gate(x_m))
        x_f = g * x_m + (1 - g) * x_in
        return self.norm(x_f), self.drop(x_f), e                     


class HCLNet(nn.Module):

    def __init__(self, d_in: int, hidden: int = 64, layers: int = 4,
                 heads: int = 4, dropout: float = 0.2, proj_dim: int = 64):
        super().__init__()
        dims = [hidden] * (layers + 1)
        self.layers = nn.ModuleList([MGHAMLayer(d_in if i == 0 else dims[i], dims[i + 1],
                                                heads, dropout) for i in range(layers)])

        self.g_phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, proj_dim))
        self.g_psi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, proj_dim))

        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, hg, return_layers: bool = True):
        h = x
        xs, es = [], []
        for layer in self.layers:
            h, _, e = layer(h, hg)
            if return_layers:
                xs.append(h)         
                es.append(e)         
        logits = self.cls(h)
        return logits, h, xs, es


def augment(x: torch.Tensor, hg: dict, mask_rate: float = 0.2, noise: float = 0.1,
            edge_rate: float = 0.1):
    m = (torch.rand_like(x) > mask_rate).float() / (1 - mask_rate)
    x_aug = x * m + noise * torch.randn_like(x)                 
    keep = (torch.rand(hg["n_edges"], device=x.device) > edge_rate)
    keep_inc = keep[hg["edge"]]
    hg_aug = dict(hg)
    hg_aug["node"] = hg["node"][keep_inc]
    hg_aug["edge"] = hg["edge"][keep_inc]                       
    return x_aug, hg_aug


def infonce(a: torch.Tensor, b: torch.Tensor, tau: float = 0.8,
            max_items: int | None = None, generator=None):
    if max_items is not None and a.shape[0] > max_items:
        sel = torch.randperm(a.shape[0], generator=generator, device=a.device)[:max_items]
        a, b = a[sel], b[sel]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def contrastive_loss(model: HCLNet, x, hg, tau=0.8, max_nodes=2048, max_edges=256):

    with torch.no_grad():
        _, _, xs, es = model(x, hg)
    total = 0.0
    gen = torch.Generator(device=x.device).manual_seed(0)
    for t in range(2):
        x_aug, hg_aug = augment(x, hg)
        _, _, xs_a, es_a = model(x_aug, hg_aug)
        for l in range(len(xs)):
            z = model.g_phi(xs[l])                              
            z_a = model.g_phi(xs_a[l])
            total = total + infonce(z, z_a, tau, max_nodes, gen)   
            w = model.g_psi(es[l])                             
            w_a = model.g_psi(es_a[l])
            total = total + infonce(w, w_a, tau, max_edges, gen)   
    return total / (2 * len(xs))                               


def class_weighted_ce(logits, y, weight=None):
    return F.cross_entropy(logits, y, weight=weight)
