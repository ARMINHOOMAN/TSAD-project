"""Data loading + a synthetic multivariate generator.
Two sources are supported:
  * "smd" -- the Server Machine Dataset (OmniAnomaly release).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import make_windows


def generate_synthetic(n_features=10, train_len=6000, test_len=3000,
                       anomaly_ratio=0.05, seed=0):
    """Correlated sinusoids + noise. Anomalies are injected only in the test
    split as spikes, level shifts and short frequency bursts."""
    rng = np.random.default_rng(seed)

    def base(T):
        t = np.arange(T)
        n_latent = 3
        freqs = rng.uniform(0.01, 0.05, size=n_latent)
        phases = rng.uniform(0, 2 * np.pi, size=n_latent)
        latents = np.stack([np.sin(2 * np.pi * f * t + p)
                            for f, p in zip(freqs, phases)], axis=1)
        mix = rng.normal(size=(n_latent, n_features))
        x = latents @ mix
        x += 0.15 * rng.normal(size=(T, n_features))
        return x

    train = base(train_len)
    test = base(test_len)
    labels = np.zeros(test_len, dtype=np.int64)

    n_anom = int(anomaly_ratio * test_len)
    placed = 0
    while placed < n_anom:
        start = rng.integers(50, test_len - 50)
        seg = rng.integers(5, 25)        
        dims = rng.choice(n_features, size=rng.integers(1, n_features), replace=False)
        kind = rng.integers(0, 3)
        if kind == 0:           
            test[start:start + seg][:, dims] += rng.normal(4, 1, size=(seg, len(dims)))
        elif kind == 1:                   
            test[start:start + seg][:, dims] += rng.uniform(2, 4)
        else:                               
            tt = np.arange(seg)
            test[start:start + seg][:, dims] += 3 * np.sin(2 * np.pi * 0.4 * tt)[:, None]
        labels[start:start + seg] = 1
        placed += seg

    return train.astype(np.float32), test.astype(np.float32), labels


def load_smd(root, entity):
    """Load one SMD entity. Expects the OmniAnomaly layout:
        root/train/<entity>.txt, root/test/<entity>.txt, root/test_label/<entity>.txt
    """
    def _read(sub):
        path = os.path.join(root, sub, entity + ".txt")
        return np.genfromtxt(path, delimiter=",", dtype=np.float32)

    train = _read("train")
    test = _read("test")
    labels = np.genfromtxt(os.path.join(root, "test_label", entity + ".txt"),
                           delimiter=",", dtype=np.int64)
    if labels.ndim > 1:
        labels = labels.max(axis=1)
    return train, test, labels.astype(np.int64)


def zscore_fit(train):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def get_data(cfg):
    """Returns normalized (train, test, labels) using train statistics only."""
    d = cfg.data
    if d.name == "synthetic":
        train, test, labels = generate_synthetic(
            n_features=d.n_features, seed=d.seed)
    elif d.name == "smd":
        train, test, labels = load_smd(d.smd_root, d.smd_entity)
    else:
        raise ValueError(f"unknown dataset {d.name}")

    mu, sd = zscore_fit(train)
    train = (train - mu) / sd
    test = (test - mu) / sd
    return train, test, labels


class WindowDataset(Dataset):
    """Wraps a (T, D) series into overlapping windows for training."""
    def __init__(self, series, L, stride):
        self.windows, self.starts = make_windows(series, L, stride)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i])
