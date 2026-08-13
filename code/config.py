"""Central configuration for the TSAD diffusion study.
"""
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    name: str = "smd"            
    smd_root: str = "../data/SMD"  
    smd_entity: str = "machine-1-1"
    window: int = 64               
    stride: int = 8               
    n_features: int = 10          
    seed: int = 0


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    latent_dim: int = 16          


@dataclass
class DiffusionConfig:
    T: int = 100                  
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    infer_steps: int = 10         
    infer_t_frac: float = 0.5      
    mask_ratio: float = 0.2        
    n_impute_masks: int = 4       


@dataclass
class ImConfig:
    window: int = 100         
    split: int = 10            
    channels: int = 64
    layers: int = 4
    nheads: int = 8
    diff_emb: int = 128
    feature_emb: int = 16
    time_emb: int = 128
    T: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.5      
    ensemble_steps: int = 30   
    ensemble_stride: int = 3   
    last_step_threshold: float = 0.02  
    unconditional: bool = True 

@dataclass
class AFConfig:
    window: int = 100          
    channels: int = 32         
    layers: int = 4            
    nheads: int = 8            
    diff_emb: int = 128
    feature_emb: int = 16
    time_emb: int = 128
    T: int = 50                
    reverse_steps: int = 50    
    beta_start: float = 1e-4
    beta_end: float = 0.01     
    mask_p: float = 0.5        

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"            


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diff: DiffusionConfig = field(default_factory=DiffusionConfig)
    im: ImConfig = field(default_factory=ImConfig)
    af: AFConfig = field(default_factory=AFConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
