# Results (smd_machine-1-1)

device=cuda, window=64, epochs=50, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.6066 | 0.5895 | 0.6247 | 0.9983 | 0.8963 | 0.5672 | 65542 | 79.6910 | 1.0808 |
| DDPM-vanilla | 0.6015 | 0.5475 | 0.6674 | 0.9994 | 0.9322 | 0.6143 | 84454 | 71.5414 | 10.8158 |
| DDPM-selective | 0.6577 | 0.5466 | 0.8255 | 0.9978 | 0.9566 | 0.6913 | 84454 | 71.7845 | 4.4001 |
| ImDiffusion | 0.3155 | 0.7958 | 0.1967 | 0.9855 | 0.5958 | 0.2392 | 447393 | -48701.4747 | 94.7952 |

