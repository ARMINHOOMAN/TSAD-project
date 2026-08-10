"""Central configuration for the diffusion-TSAD study.

Everything that changes across experiments lives here so the scripts stay clean.
Values are intentionally small so the whole pipeline runs on a CPU in minutes;
bump epochs / d_model / T for the real report runs.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    name: str = "smd"              # "synthetic" or "smd"
    smd_root: str = "../data/SMD"  # only used when name == "smd"
    smd_entity: str = "machine-1-1"
    window: int = 64               # sliding window length L
    stride: int = 8                # stride for training windows (test uses stride=1)
    n_features: int = 10           # only used by the synthetic generator
    seed: int = 0


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    latent_dim: int = 16           # LSTM-VAE latent size


@dataclass
class DiffusionConfig:
    T: int = 100                   # training diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    infer_steps: int = 10          # DDIM steps used at scoring time (cheap)
    infer_t_frac: float = 0.5      # start denoising from this fraction of T
    mask_ratio: float = 0.2        # selective-denoising: fraction of noised elements
    n_impute_masks: int = 4        # masking mode: interleaved temporal masks


@dataclass
class ImConfig:
    """Faithful ImDiffusion (Chen et al., 2023) settings. Defaults follow the
    paper/official repo (config/base.yaml); shrink them for CPU smoke runs."""
    window: int = 100          # paper detection window
    split: int = 10            # grating blocks -> 5 masked + 5 unmasked (paper Table 1)
    channels: int = 64
    layers: int = 4
    nheads: int = 8
    diff_emb: int = 128
    feature_emb: int = 16
    time_emb: int = 128
    T: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.5      # quad schedule
    ensemble_steps: int = 30   # keep the LAST 30 denoising steps ...
    ensemble_stride: int = 3   # ... every 3rd -> range(0,30,3) = 10 votes
                               # (paper Sec 4.5 / ensemble_proper: "sample every 3
                               #  steps from the last 30 denoising steps". The
                               #  reverse loop always runs all T=50 steps; these
                               #  two only choose which ones cast a vote.)
    last_step_threshold: float = 0.02  # tau_T in the adaptive per-step threshold
    unconditional: bool = True # paper Sec 4.1: the observed region is fed as its
                               # forward NOISE, not its clean values. The official
                              # repo runs this way (exe_machine: unconditional_list
                               # = [True]) even though base.yaml defaults to 0.


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"            # auto-set to cuda if available in run script


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diff: DiffusionConfig = field(default_factory=DiffusionConfig)
    im: ImConfig = field(default_factory=ImConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
