# Results (synthetic)

device=cuda, window=64, epochs=2, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.6111 | 0.7273 | 0.5269 | 0.8988 | 0.8691 | 0.6201 | 55270 | 1.7850 | 0.0851 |
| DDPM-vanilla | 0.5934 | 0.7642 | 0.4850 | 0.8883 | 0.9087 | 0.6483 | 80326 | 1.1193 | 0.4499 |
| DDPM-selective | 0.5972 | 0.7107 | 0.5150 | 0.8704 | 0.9190 | 0.6617 | 80326 | 0.9961 | 0.1852 |
| ImDiffusion | 0.2543 | 0.2984 | 0.2216 | 0.6805 | 0.5948 | 0.1033 | 50161 | 1.6600 | 0.6425 |

