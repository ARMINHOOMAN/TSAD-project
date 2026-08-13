"""Implementation of AnomalyFilter (Obata et al., 2026).

The method is two components that only work together:

  * Masked Gaussian noise
  * Noiseless inference
"""
import time

import numpy as np
import torch
import torch.nn as nn

from imdiffusion import DiffCSDI
from utils import make_windows, windows_to_series


class AnomalyFilter(nn.Module):
    def __init__(self, n_features, cfg, device="cpu"):
        super().__init__()
        c = cfg.af
        self.K = n_features
        self.L = c.window
        self.T = c.T
        self.lam = min(c.reverse_steps, c.T)
        self.p = c.mask_p
        self.time_emb, self.feat_emb = c.time_emb, c.feature_emb

        side_dim = c.time_emb + c.feature_emb
        self.embed_layer = nn.Embedding(n_features, c.feature_emb)
        self.diffmodel = DiffCSDI(side_dim, c.channels, c.layers, c.nheads,
                                  c.diff_emb, c.T, inputdim=1)

        beta = np.linspace(c.beta_start, c.beta_end, c.T)
        self.beta = beta
        self.alpha_hat = 1 - beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("alpha_torch",
                             torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1))

    # -- (CSDI) ---------------------------------------------
    def _time_embedding(self, L, device):
        pos = torch.arange(L, device=device).float().unsqueeze(0)
        pe = torch.zeros(1, L, self.time_emb, device=device)
        div = 1 / torch.pow(10000.0, torch.arange(0, self.time_emb, 2,
                                                  device=device) / self.time_emb)
        pe[:, :, 0::2] = torch.sin(pos.unsqueeze(2) * div)
        pe[:, :, 1::2] = torch.cos(pos.unsqueeze(2) * div)
        return pe

    def side_info(self, B, device):
        te = self._time_embedding(self.L, device).unsqueeze(2).expand(B, -1, self.K, -1)
        fe = self.embed_layer(torch.arange(self.K, device=device))
        fe = fe.unsqueeze(0).unsqueeze(0).expand(B, self.L, -1, -1)
        return torch.cat([te, fe], dim=-1).permute(0, 3, 2, 1)

    def _strat(self, B, device):
        return torch.zeros(B, device=device, dtype=torch.long)

    # -- masked Gaussian noise --------------------------------------
    def masked_noise(self, x0):
        zeta = torch.randn_like(x0)
        bern = (torch.rand_like(x0) < self.p).float()
        return bern * zeta

    # -- training loss ------------------------------------------
    def loss(self, x0):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        abar = self.alpha_torch[t]
        eps = self.masked_noise(x0)
        xt = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
        pred = self.diffmodel(xt.unsqueeze(1), self.side_info(B, device), t,
                              self._strat(B, device))
        return (eps - pred).abs().mean()

    # -- noiseless inference, Algorithm 4 ------------------------------------
    @torch.no_grad()
    def reconstruct(self, x0):
        B = x0.shape[0]
        device = x0.device
        side = self.side_info(B, device)
        strat = self._strat(B, device)

        xhat = float(self.alpha[self.lam - 1] ** 0.5) * x0

        for t in range(self.lam - 1, -1, -1):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.diffmodel(xhat.unsqueeze(1), side, tt, strat)
            c1 = 1 / self.alpha_hat[t] ** 0.5
            c2 = self.beta[t] / (1 - self.alpha[t]) ** 0.5
            xhat = c1 * (xhat - c2 * eps)
        return xhat


# --------------------------------------------------------------------------
# Training + scoring
# --------------------------------------------------------------------------
def train_anomalyfilter(model, train_series, cfg, device):
    L = model.L
    windows, _ = make_windows(train_series, L, cfg.data.stride)
    ds = torch.from_numpy(windows)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    model.train()
    t0 = time.time()

    for ep in range(cfg.train.epochs):
        order = torch.randperm(len(ds))
        running, nb = 0.0, 0
        for i in range(0, len(ds), cfg.train.batch_size):
            xb = ds[order[i:i + cfg.train.batch_size]].to(device).permute(0, 2, 1)
            loss = model.loss(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        print(f"    epoch {ep + 1:2d}/{cfg.train.epochs}  loss={running / nb:.4f}")

    return time.time() - t0


@torch.no_grad()
def score_anomalyfilter(model, test_series, cfg, device):
    """MSE between input and reconstruction, summed over features, then smoothed
    with a moving average of half the window (repo: np.convolve, half_window=50)."""
    L = model.L
    T_total = len(test_series)
    windows, starts = make_windows(test_series, L, L)
    ds = torch.from_numpy(windows).to(device)

    t0 = time.time()
    errs = []
    for i in range(0, len(ds), cfg.train.batch_size):
        xb = ds[i:i + cfg.train.batch_size].permute(0, 2, 1)        
        recon = model.reconstruct(xb)
        errs.append(((xb - recon) ** 2).sum(dim=1).cpu().numpy()) 
    infer_time = time.time() - t0

    win_scores = np.concatenate(errs, axis=0)                       
    score = windows_to_series(win_scores, starts, T_total, L)

    hw = max(1, L // 2)          
    score = np.convolve(score, np.ones(hw) / hw, mode="same")
    return score.astype(np.float64), infer_time
