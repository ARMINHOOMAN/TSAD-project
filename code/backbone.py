"""Transformer denoiser used by all three diffusion regimes.
"""
import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64, n_heads=4, n_layers=2,
                 ff_dim=128, dropout=0.1, max_len=512):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, x, t):
        # x: (B, L, in_dim), t: (B,)
        L = x.size(1)
        h = self.in_proj(x) + self.pos[:, :L]
        h = h + self.t_embed(t)[:, None, :]
        h = self.encoder(h)
        return self.out(h)
