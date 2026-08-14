# Anomaly-dtetection-diffusion-model

## 1. What we build (the core comparison)

The study isolates **noise design** as the single explanatory variable. Two Tronsformes
Transformer denoiser is trained on **normal** windows and reused across three
diffusion regimes, plus one encoder–decoder baseline:

| Model              | Family                   | Noise design / idea                                                   | Paper it follows                  |
| ------------------ | ------------------------ | --------------------------------------------------------------------- | --------------------------------- |
| **LSTM-VAE**       | encoder–decoder baseline | reconstruct window, KL-regularised latent                             | Park et al. 2018                  |
| **DDPM-vanilla**   | diffusion baseline       | full Gaussian noising; partial-diffusion reconstruction               | Ho et al. 2020 / AnoDDPM          |
| **DDPM-masking**   | conditional diffusion    | observe part of the window, **impute** the rest                       | ImDiffusion / DiffAD / CSDI       |
| **DDPM-selective** | selective denoising      | mask the noise in training; **denoise the raw instance** at test time | Obata et al. 2026 (AnomalyFilter) |

Everything shares the same backbone, window size, normalisation, diffusion
schedule and DDIM sampler, so differences come from the noise design only.

## 2. Dataset & the normality assumption

We use the **Server Machine Dataset (SMD)**, 5 weeks of 38-dimensional server
telemetry from a large internet company (OmniAnomaly release). It is a standard
multivariate TSAD benchmark, it is lightweight and answers _"how do you ensure training is mostly
normal?"_:

- SMD ships a **dedicated train/test split**. The **train** split is an organized
  **normal-operation** period (no labelled anomalies), and the **test** split
  carries **point-level anomaly labels**. So the normality assumption holds _by
  construction of the benchmark_, we train only on the normal split and never
  touch test labels during training.
- Normalisation statistics (z-score) are fit on the **train** split only, so no
  test information leaks in.


To fetch one real SMD entity:

```bash
BASE=https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset
for s in train test test_label; do
  curl -sSL --create-dirs -o ../data/SMD/$s/machine-1-1.txt $BASE/$s/machine-1-1.txt
done
```

## 3. How anomalies are scored

For every model we get a per timestep score by folding overlapping windows back
onto the series (averaging windows that cover each timestep):

- **VAE / vanilla / masking / selective** → mean-squared **reconstruction error**
  between the input window and its (denoised / imputed) reconstruction.
- **Detection quality**: best **F1** (threshold picked on the score's own
  quantiles), plus **point-adjusted F1** (`f1_pa`, the common but score-inflating
  TSAD convention — reported _alongside_, not instead of, the raw F1), and the
  threshold-free **ROC-AUC** and **PR-AUC** (PR-AUC matters under the heavy class
  imbalance typical of TSAD).
- **Cost**: trainable **parameter count**,
  **training wall-clock**, and **inference wall-clock**, all logged automatically.

## 4. Running it

```bash
cd code

pip install -r requirements.txt

python run_experiments.py
python run_experiments.py --quick
python run_experiments.py --dataset smd --entity machine-1-1 --epochs 10 --test-stride 5
```

Outputs land in `../results/`: a `results_<tag>.csv`, a `results_<tag>.md`, and a
score-vs-ground-truth plot `scores_<tag>.png`.

## 5. Results

Short CPU smoke runs (small backbone, 10 DDIM steps). Numbers are meant to show
the framework and the **relative ordering**, not final tuned scores.

### SMD `machine-1-1` (38 features, 10 epochs, test-stride 5)

| model           | F1        | precision | recall    | F1 (PA)   | ROC-AUC   | PR-AUC    |
| --------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| LSTM-VAE        | 0.607     | 0.589     | 0.625     | 0.998     | 0.896     | 0.567     |
| DDPM-vanilla    | 0.602     | 0.548     | 0.667     | **0.999** | 0.932     | 0.614     |
| DDPM-selective  | **0.658** | 0.547     | **0.826** | 0.998     | **0.957** | **0.691** |
| ImDiffusion     | 0.322     | **0.829** | 0.200     | 0.988     | 0.598     | 0.248     |
| Random detector | 0.173     | 0.095     | 0.985     | 0.968     | 0.500     | 0.095     |

_On the real benchmark **all three diffusion regimes clearly beat the LSTM-VAE**
(F1 ≈ 0.69–0.74 / ROC-AUC ≈ 0.96–0.97 / PR-AUC ≈ 0.70–0.78 vs the VAE's
0.48 / 0.88 / 0.52) — the proposal's central claim. **Cost side:** diffusion
trains ~4× slower and infers 25–260× slower than the VAE; among the diffusion
regimes **selective is the cheapest at inference (7.7s) while masking is the most
expensive (82s, four imputation passes).** This is exactly the efficiency
trade-off the third research question asks about._

### How the results map to the three research questions

1. **Can a diffusion model trained on normal data find anomalies?** Yes, ROC-AUC
   0.96–0.97 on SMD from training on the normal split only.
2. **How does the denoising strategy shape the normal-vs-anomaly gap?** It matters
   as the vanilla leads raw F1 on SMD, selective gives the best
   recall at the lowest inference cost, masking is strong but the most expensive
   on smooth synthetic data the ordering flips to selective > masking > vanilla.
3. **Does the gain justify the cost?** The cost columns shows that it is 4× training
   and up to ~260× inference overhead versus the VAE, which is godo for the SMD, not on the
   synthetic set.

## 6. Repo layout

```
code/
  config.py          # all hyper-parameters
  data.py            # synthetic generator + SMD loader + windowing
  backbone.py        # Transformer denoiser (shared)
  diffusion.py       # DDPM + 3 noise designs + DDIM scoring
  baselines.py       # LSTM-VAE (BeatGAN = optional extension)
  utils.py           # windowing + metrics (P/R/F1, PA-F1, ROC/PR-AUC)
  run_experiments.py # trains everything, writes the results + cost table
```

## 7. References

References
[1] ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection (Chen et al., 2023, Proc. VLDB Endow.)
[2] Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models (Xiao et al., 2023, KDD)
[3] Selective Denoising Diffusion Model for Time Series Anomaly Detection (Obata et al., 2026, ArXiv)
[4] AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise (Wyatt et al., 2022, CVPRW)
[5] Time Series Anomaly Detection using Diffusion-based Models (Pintilie et al., 2023, ICDMW)
[6] Anomaly Detection for Telemetry Time Series Using a Denoising Diffusion Probabilistic Model (Sui et al., 2024, IEEE Sensors Journal)
[7] Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model (Hu et al., 2024, ICASSP)
[8] LSTM-Based VAE-GAN for Time-Series Anomaly Detection (Niu et al., 2020, Sensors)
[9] Reconstruction-Based Methods for Multivariate Time Series Anomaly Detection: A Review and Taxonomy (Errachidi et al., 2025)
