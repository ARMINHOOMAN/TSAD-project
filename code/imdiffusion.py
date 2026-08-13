"""Implementation of ImDiffusion (Chen et al., VLDB 2023).

  * grating mask  : window split into blocks
  * backbone      : CSDI = per-block temporal + feature transformers, with a
                    mask-index (strategy) embedding
  * diffusion     : T steps, quad beta schedule
  * anomaly score : ancestral sampling captures denoising steps for the last
                    ~10 steps compute residual = sum_feat, then VOTE.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_windows, windows_to_series


# --------------------------------------------------------------------------
# CSDI backbone (ported from diff_models.py)
# --------------------------------------------------------------------------
def get_torch_trans(heads=8, layers=1, channels=64):
    layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads,
                                       dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=layers)


def conv1d_init(in_c, out_c, k=1):
    layer = nn.Conv1d(in_c, out_c, k)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, dim=128):
        super().__init__()
        self.register_buffer("embedding", self._table(num_steps, dim // 2),
                             persistent=False)
        self.p1 = nn.Linear(dim, dim)
        self.p2 = nn.Linear(dim, dim)

    def _table(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        freqs = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, t):
        x = self.embedding[t]
        x = F.silu(self.p1(x))
        x = F.silu(self.p2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diff_emb, nheads):
        super().__init__()
        self.diff_proj = nn.Linear(diff_emb, channels)
        self.strat_proj = nn.Linear(diff_emb, channels)
        self.cond_proj = conv1d_init(side_dim, 2 * channels)
        self.mid_proj = conv1d_init(channels, 2 * channels)
        self.out_proj = conv1d_init(channels, 2 * channels)
        self.time_layer = get_torch_trans(nheads, 1, channels)
        self.feat_layer = get_torch_trans(nheads, 1, channels)

    def _time(self, y, shape):
        B, C, K, L = shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)

    def _feat(self, y, shape):
        B, C, K, L = shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feat_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)

    def forward(self, x, cond_info, diff_emb, strat_emb):
        B, C, K, L = x.shape
        shape = x.shape
        x = x.reshape(B, C, K * L)
        d = self.diff_proj(diff_emb).unsqueeze(-1)
        s = self.strat_proj(strat_emb).unsqueeze(-1)
        y = x + d + s
        y = self._time(y, shape)
        y = self._feat(y, shape)
        y = self.mid_proj(y)
        _, cdim, _, _ = cond_info.shape
        y = y + self.cond_proj(cond_info.reshape(B, cdim, K * L))
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.out_proj(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(shape)
        return (x + residual.reshape(shape)) / math.sqrt(2.0), skip.reshape(shape)


class DiffCSDI(nn.Module):
    def __init__(self, side_dim, channels, layers, nheads, diff_emb, num_steps, inputdim=2):
        super().__init__()
        self.channels = channels
        self.diff_embed = DiffusionEmbedding(num_steps, diff_emb)
        self.strat_embed = nn.Embedding(2, diff_emb)          # grating mask index p
        self.in_proj = conv1d_init(inputdim, channels)
        self.out1 = conv1d_init(channels, channels)
        self.out2 = conv1d_init(channels, 1)
        nn.init.zeros_(self.out2.weight)
        self.blocks = nn.ModuleList([
            ResidualBlock(side_dim, channels, diff_emb, nheads) for _ in range(layers)])

    def forward(self, x, cond_info, t, strat):
        B, inputdim, K, L = x.shape
        x = self.in_proj(x.reshape(B, inputdim, K * L))
        x = F.relu(x).reshape(B, self.channels, K, L)
        de = self.diff_embed(t)
        se = self.strat_embed(strat)
        skips = []
        for blk in self.blocks:
            x, skip = blk(x, cond_info, de, se)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.blocks))
        x = x.reshape(B, self.channels, K * L)
        x = F.relu(self.out1(x))
        x = self.out2(x).reshape(B, K, L)
        return x


# --------------------------------------------------------------------------
# ImDiffusion model (grating mask + conditional diffusion, from main_model.py)
# --------------------------------------------------------------------------
class ImDiffusion(nn.Module):
    def __init__(self, n_features, cfg, device="cpu"):
        super().__init__()
        c = cfg.im
        self.K = n_features
        self.L = c.window
        self.split = c.split
        self.T = c.T
        self.uncond = c.unconditional
        self.time_emb, self.feat_emb = c.time_emb, c.feature_emb
        # unconditional: 1 input channel and no conditional-mask side channel
        # (main_model.CSDI_base: input_dim = 1 if is_unconditional else 2)
        side_dim = c.time_emb + c.feature_emb + (0 if self.uncond else 1)
        self.embed_layer = nn.Embedding(n_features, c.feature_emb)
        self.diffmodel = DiffCSDI(side_dim, c.channels, c.layers, c.nheads,
                                  c.diff_emb, c.T, inputdim=1 if self.uncond else 2)

        beta = np.linspace(c.beta_start ** 0.5, c.beta_end ** 0.5, c.T) ** 2   # quad
        self.beta = beta
        self.alpha_hat = 1 - beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("alpha_torch",
                             torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1))

    # -- grating mask (dataset.py get_mask) --
    def grating_mask(self, B, strategy, device):
        m = torch.zeros(self.L, device=device)
        skip = max(1, self.L // self.split)
        for bi, beg in enumerate(range(0, self.L, skip)):
            observe = (bi % 2 == 0) if strategy == 0 else (bi % 2 != 0)
            if observe:
                m[beg:min(beg + skip, self.L)] = 1
        return m.view(1, 1, self.L).expand(B, self.K, self.L).contiguous()

    def _time_embedding(self, L, device):
        pos = torch.arange(L, device=device).float().unsqueeze(0)
        pe = torch.zeros(1, L, self.time_emb, device=device)
        div = 1 / torch.pow(10000.0, torch.arange(0, self.time_emb, 2, device=device) / self.time_emb)
        pe[:, :, 0::2] = torch.sin(pos.unsqueeze(2) * div)
        pe[:, :, 1::2] = torch.cos(pos.unsqueeze(2) * div)
        return pe 

    def side_info(self, B, cond_mask):
        device = cond_mask.device
        te = self._time_embedding(self.L, device).unsqueeze(2).expand(B, -1, self.K, -1)
        fe = self.embed_layer(torch.arange(self.K, device=device))
        fe = fe.unsqueeze(0).unsqueeze(0).expand(B, self.L, -1, -1)
        info = torch.cat([te, fe], dim=-1).permute(0, 3, 2, 1) 
        if not self.uncond: 
            info = torch.cat([info, cond_mask.unsqueeze(1)], dim=1)
        return info

    def _model_input(self, noisy, x0, cond_mask):
        if self.uncond:
            return noisy.unsqueeze(1) 
        cond_obs = (cond_mask * x0).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    # -- training loss ------
    def loss(self, x0, strategy):
        B = x0.shape[0]
        device = x0.device
        cond_mask = self.grating_mask(B, strategy, device)
        t = torch.randint(0, self.T, (B,), device=device)
        abar = self.alpha_torch[t]
        noise = torch.randn_like(x0)
        noisy = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
        inp = self._model_input(noisy, x0, cond_mask)
        strat = torch.full((B,), strategy, device=device, dtype=torch.long)
        pred = self.diffmodel(inp, self.side_info(B, cond_mask), t, strat)
        target_mask = 1 - cond_mask
        resid = (noise - pred) * target_mask
        return (resid ** 2).sum() / target_mask.sum().clamp(min=1)

    # -- ancestral sampling -------
    @torch.no_grad()
    def impute_middle(self, x0, strategy, keep_steps):
        B = x0.shape[0]
        device = x0.device
        cond_mask = self.grating_mask(B, strategy, device)
        side = self.side_info(B, cond_mask)
        strat = torch.full((B,), strategy, device=device, dtype=torch.long)

        if self.uncond:
            noisy_obs = x0
            noisy_hist = []
            for t in range(self.T):
                noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs \
                            + (self.beta[t] ** 0.5) * torch.randn_like(noisy_obs)
                noisy_hist.append(noisy_obs * cond_mask)
        else:
            cond_obs = (cond_mask * x0).unsqueeze(1)

        sample = torch.randn_like(x0)
        middle = {}
        for t in range(self.T - 1, -1, -1):
            if self.uncond:
                inp = (cond_mask * noisy_hist[t]
                       + (1 - cond_mask) * sample).unsqueeze(1)
            else:
                noisy_target = ((1 - cond_mask) * sample).unsqueeze(1)
                inp = torch.cat([cond_obs, noisy_target], dim=1)
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            pred = self.diffmodel(inp, side, tt, strat)
            c1 = 1 / self.alpha_hat[t] ** 0.5
            c2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            sample = c1 * (sample - c2 * pred)
            if t > 0:
                sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                sample = sample + sigma * torch.randn_like(sample)
            if t in keep_steps:
                middle[t] = (cond_mask * x0 + (1 - cond_mask) * sample).detach().clone()
        return middle, cond_mask


# --------------------------------------------------------------------------
# Training + vote-ensemble scoring
# --------------------------------------------------------------------------
def train_imdiffusion(model, train_series, cfg, device):
    import time
    L = model.L
    windows, _ = make_windows(train_series, L, cfg.data.stride)
    ds = torch.from_numpy(windows)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    model.train()
    t0 = time.time()
    for ep in range(cfg.train.epochs):
        perm = torch.randperm(len(ds))
        running, nb = 0.0, 0
        for i in range(0, len(ds), cfg.train.batch_size):
            xb = ds[perm[i:i + cfg.train.batch_size]].to(device)
            xb = xb.permute(0, 2, 1)
            strategy = 0 if torch.rand(1).item() < 0.5 else 1
            loss = model.loss(xb, strategy)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        print(f"    epoch {ep + 1:2d}/{cfg.train.epochs}  loss={running / nb:.4f}")
    return time.time() - t0


@torch.no_grad()
def score_imdiffusion(model, test_series, cfg, device, return_details=False):
    """ImDiffusion vote-ensemble score, folded to one value per timestep.

    The reverse process always runs all T steps; `keep` selects which of them cast
    a vote -- range(0, ensemble_steps, ensemble_stride) counted back from the
    fully-denoised end (t=0). With return_details=True a dict of per-step
    diagnostics is also returned, for the ensemble-inference figure.
    """
    import time
    L = model.L
    T_total = len(test_series)
    c = cfg.im
    keep = [s for s in range(0, c.ensemble_steps, c.ensemble_stride) if s < model.T]
    windows, starts = make_windows(test_series, L, L)
    ds = torch.from_numpy(windows).to(device)
    B_all = len(ds)

    t0 = time.time()
    imp = {s: np.zeros((B_all, L, model.K), dtype=np.float32) for s in keep}
    for strategy in (0, 1):
        cm = model.grating_mask(1, strategy, device)[0, 0].cpu().numpy()
        target_pos = cm < 0.5
        for i in range(0, B_all, cfg.train.batch_size):
            xb = ds[i:i + cfg.train.batch_size].permute(0, 2, 1)
            middle, _ = model.impute_middle(xb, strategy, set(keep))
            for s in keep:
                m = middle[s].permute(0, 2, 1).cpu().numpy()
                imp[s][i:i + xb.shape[0]][:, target_pos, :] = m[:, target_pos, :]
    infer_time = time.time() - t0

    x_win = windows
    step_resid = {}
    avgE = {}
    for s in keep:
        r_win = ((imp[s] - x_win) ** 2).sum(axis=-1)
        step_resid[s] = windows_to_series(r_win, starts, T_total, L) 
        avgE[s] = float(r_win.mean())

    N = T_total
    e0 = avgE[keep[0]]
    votes = np.zeros(N, dtype=np.float32)
    step_thr, step_pred = {}, {}
    for s in keep:
        proper = e0 * c.last_step_threshold / max(avgE[s], 1e-12)
        k = max(int(proper * N), 1)
        r = step_resid[s]
        thr = np.partition(r, N - k)[N - k]
        y = (r >= thr).astype(np.float32)
        votes += y
        step_thr[s], step_pred[s] = float(thr), y

    if return_details:
        details = dict(keep=keep, votes=votes, step_resid=step_resid,
                       step_thr=step_thr, step_pred=step_pred, avgE=avgE,
                       imp=imp, windows=windows, starts=starts, L=L, T=model.T,
                       n_votes=len(keep))
        return votes, infer_time, details
    return votes, infer_time
