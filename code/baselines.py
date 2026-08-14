"""Reconstruction baselines. LSTM-VAE is fully implemented as the encoder-decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVAE(nn.Module):
    def __init__(self, n_features, hidden=64, latent=16, num_layers=1):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, num_layers, batch_first=True)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        B, L, D = x.shape
        _, (h, _) = self.enc(x)
        h = h[-1]                       
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h0 = self.from_z(z)[None].expand(1, B, -1).contiguous()
        c0 = torch.zeros_like(h0)
        seq = h0[-1][:, None, :].expand(B, L, -1)  
        dec_out, _ = self.dec(seq, (h0, c0))
        return self.out(dec_out), mu, logvar

    def loss(self, x, beta=1.0):
        recon, mu, logvar = self(x)
        rec = F.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + beta * kld

    @torch.no_grad()
    def score_windows(self, x):
        recon, _, _ = self(x)
        return ((x - recon) ** 2).mean(dim=-1)  
