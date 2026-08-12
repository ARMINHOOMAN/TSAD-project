# Results (smd_machine-1-1)

device=cuda, window=64, epochs=10, diffusion T=100, infer_steps=10

| model | f1 | precision | recall | f1_pa | roc_auc | pr_auc | threshold | params | train_s | infer_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM-VAE | 0.5704 | 0.5192 | 0.6329 | 0.9983 | 0.8892 | 0.5459 | 5.9867 | 65542 | 1.8941 | 0.1498 |
| DDPM-vanilla | 0.6941 | 0.6200 | 0.7884 | 0.9963 | 0.9661 | 0.7113 | 0.2040 | 84454 | 2.7274 | 3.2303 |
| AnomalyFilter | 0.6557 | 0.6918 | 0.6232 | 0.9627 | 0.8296 | 0.5917 | 0.5833 | 190433 | 280.7713 | 30.4946 |

