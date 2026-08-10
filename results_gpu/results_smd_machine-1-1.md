# Results (smd_machine-1-1)

device=cuda, window=64, epochs=50, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.6127 | 0.5883 | 0.6392 | 0.9983 | 0.8977 | 0.5718 | 65542 | 8.6882 | 0.0304 |
| DDPM-vanilla | 0.5970 | 0.5489 | 0.6544 | 0.9987 | 0.9250 | 0.5986 | 84454 | 13.4621 | 0.6719 |
| DDPM-selective | 0.6296 | 0.5232 | 0.7903 | 0.9980 | 0.9449 | 0.6471 | 84454 | 13.8250 | 0.3235 |
| ImDiffusion | 0.3229 | 0.7460 | 0.2060 | 0.9893 | 0.5996 | 0.2475 | 112897 | 390.1131 | 36.4724 |

