"""DDPM for time-series anomaly detection with three interchangeable noise
designs -- this is the "explanatory variable" the study isolates:

  * "vanilla"   -- plain DDPM. Score by partial diffusion: noise the window to a
                   middle timestep, denoise back, compare (AnoDDPM-style).
  * "masking"   -- conditional imputation (ImDiffusion / DiffAD / CSDI-style):
                   observe part of the window, impute the rest, compare.
  * "selective" -- selective denoising (AnomalyFilter / Obata-style): mask the
                   Gaussian noise during training and, at inference, denoise the
                   *raw* instance without adding noise, so the model mostly edits
                   anomalous regions and leaves normal ones intact.

All three share one Transformer denoiser and DDIM sampling so the comparison is
apples-to-apples and cheap enough to run on CPU.
"""
import numpy as np
import torch
import torch.nn as nn

from backbone import Denoiser


def build_diffusion(mode, n_features, cfg):
    D = n_features
    in_dim = 3 * D if mode == "masking" else D 
    m = cfg.model
    model = Denoiser(in_dim, D, d_model=m.d_model, n_heads=m.n_heads,
                     n_layers=m.n_layers, ff_dim=m.ff_dim, dropout=m.dropout,
                     max_len=cfg.data.window)
    return GaussianDiffusion(model, mode, D, cfg)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, mode, n_features, cfg):
        super().__init__()
        self.model = model
        self.mode = mode
        self.D = n_features
        d = cfg.diff
        self.T = d.T
        self.infer_steps = d.infer_steps
        self.mask_ratio = d.mask_ratio
        self.K = d.n_impute_masks
        self.n_score_samples = 2 
        frac = {"vanilla": 0.6, "masking": 0.5, "selective": 0.3}[mode]
        self.t_start = max(1, int(frac * d.T))

        betas = torch.linspace(d.beta_start, d.beta_end, d.T)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    # -- forward diffusion ---------------------------------------------------
    def q_sample(self, x0, t, noise):
        abar = self.alpha_bar[t].view(-1, 1, 1)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    # -- training loss -------------------------------------------------------
    def p_losses(self, x0):
        B, L, D = x0.shape
        dev = x0.device
        t = torch.randint(0, self.T, (B,), device=dev)
        noise = torch.randn_like(x0)

        if self.mode == "vanilla":
            x_t = self.q_sample(x0, t, noise)
            inp, target, lmask = x_t, noise, torch.ones_like(x0)

        elif self.mode == "masking":
            obs = (torch.rand(B, L, 1, device=dev) > 0.3).float()
            x_noised = self.q_sample(x0, t, noise)
            x_t = obs * x0 + (1 - obs) * x_noised
            inp = torch.cat([x_t, obs * x0, obs.expand(-1, -1, D)], dim=-1)
            target, lmask = noise, (1 - obs).expand(-1, -1, D)

        elif self.mode == "selective":
            nmask = (torch.rand_like(x0) < self.mask_ratio).float()
            x_noised = self.q_sample(x0, t, noise)
            x_t = (1 - nmask) * x0 + nmask * x_noised
            inp, target, lmask = x_t, noise, nmask
        else:
            raise ValueError(self.mode)

        pred = self.model(inp, t)
        return ((pred - target) ** 2 * lmask).sum() / (lmask.sum() + 1e-8)

    # -- DDIM helpers --------------------------------------------------------
    def _infer_seq(self):
        seq = np.linspace(self.t_start, 0, self.infer_steps + 1)
        return [int(round(s)) for s in seq]

    def _x0_from_eps(self, x_t, t_idx, eps):
        abar = self.alpha_bar[t_idx]
        return (x_t - (1 - abar).sqrt() * eps) / abar.sqrt()

    def _ddim_step(self, x0_pred, eps, t_prev):
        abar_prev = self.alpha_bar[t_prev]
        return abar_prev.sqrt() * x0_pred + (1 - abar_prev).sqrt() * eps

    # -- reconstruction / scoring -------------------------------------------
    @torch.no_grad()
    def reconstruct(self, x0):
        B, L, D = x0.shape
        dev = x0.device
        seq = self._infer_seq()

        if self.mode == "vanilla":
            x_t = self.q_sample(x0, torch.full((B,), seq[0], device=dev),
                                torch.randn_like(x0))
            x0_pred = x_t
            for i in range(len(seq) - 1):
                tc, tn = seq[i], seq[i + 1]
                eps = self.model(x_t, torch.full((B,), tc, device=dev))
                x0_pred = self._x0_from_eps(x_t, tc, eps)
                x_t = self._ddim_step(x0_pred, eps, tn)
            return x0_pred

        if self.mode == "selective":
            # anomalous parts (which look like removable noise) get dragged, so |x - x_hat|
            t = torch.full((B,), self.t_start, device=dev)
            x_t = x0.clone()
            for _ in range(self.infer_steps):
                eps = self.model(x_t, t)
                x_t = self._x0_from_eps(x_t, self.t_start, eps)
            return x_t

        # masking
        pos = torch.arange(L, device=dev)
        recon = torch.zeros_like(x0)
        count = torch.zeros(B, L, 1, device=dev)
        for k in range(self.K):
            obs = (pos % self.K != k).float().view(1, L, 1).expand(B, L, 1)
            x_t = obs * x0 + (1 - obs) * self.q_sample(
                x0, torch.full((B,), seq[0], device=dev), torch.randn_like(x0))
            x0_pred = x_t
            for i in range(len(seq) - 1):
                tc, tn = seq[i], seq[i + 1]
                inp = torch.cat([x_t, obs * x0, obs.expand(-1, -1, D)], dim=-1)
                eps = self.model(inp, torch.full((B,), tc, device=dev))
                x0_pred = self._x0_from_eps(x_t, tc, eps)
                x_prev = self._ddim_step(x0_pred, eps, tn)
                x_t = obs * x0 + (1 - obs) * x_prev
            recon += (1 - obs) * x0_pred
            count += (1 - obs)
        count = torch.clamp(count, min=1.0)
        return recon / count

    @torch.no_grad()
    def score_windows(self, x0):
        """Per-window, per-timestep anomaly score = mean squared recon error.
        Stochastic modes average a few draws; selective is deterministic."""
        n = 1 if self.mode == "selective" else self.n_score_samples
        acc = torch.zeros_like(x0)
        for _ in range(n):
            recon = self.reconstruct(x0)
            acc = acc + (x0 - recon) ** 2
        return (acc / n).mean(dim=-1)
