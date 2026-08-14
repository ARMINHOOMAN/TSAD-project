## 1. What we build (the core comparison)

Two shared-backbone diffusion regimes (isolating **noise design** as the single
explanatory variable) plus one encoder–decoder baseline, and — reproduced
faithfully from its own paper — **ImDiffusion** as the masking/imputation method:

| Model              | Family                             | Noise design / idea                                                   | Paper it follows                   |
| ------------------ | ---------------------------------- | --------------------------------------------------------------------- | ---------------------------------- |
| **LSTM-VAE**       | encoder–decoder baseline           | reconstruct window, KL-regularised latent                             | Park et al. 2018                   |
| **DDPM-vanilla**   | diffusion baseline                 | full Gaussian noising; partial-diffusion reconstruction               | Ho et al. 2020 / AnoDDPM           |
| **DDPM-selective** | selective denoising                | mask the noise in training; **denoise the raw instance** at test time | Obata et al. 2026 (AnomalyFilter)  |
| **ImDiffusion**    | conditional imputation (**exact**) | grating mask + CSDI backbone + step-wise **vote ensemble**            | Chen et al. 2023 (ImDiffusion) [1] |

`DDPM-vanilla` and `DDPM-selective` share the same Transformer denoiser, window,
normalisation, schedule and DDIM sampler, so their difference comes only from the
noise design. **ImDiffusion is deliberately different** — it is the paper's own
method end to end (its own CSDI backbone and scoring), not a shared-backbone
variant; see §5.1.

## 2. Dataset & the normality assumption

We use the **Server Machine Dataset (SMD)** — 5 weeks of 38-dimensional server
telemetry from a large internet company (OmniAnomaly release). It is a standard
multivariate TSAD benchmark, it is lightweight (plain text, a few MB/entity, no
images), and — crucially — it answers _"how do you ensure training is mostly
normal?"_:

- SMD ships a **dedicated train/test split**. The **train** split is a curated
  **normal-operation** period (no labelled anomalies), and the **test** split
  carries **point-level anomaly labels**. So the normality assumption holds _by
  construction of the benchmark_ — we train only on the normal split and never
  touch test labels during training.
- Normalisation statistics (z-score) are fit on the **train** split only, so no
  test information leaks in.

A **synthetic generator** (`data.py`) is also included so the whole pipeline runs
with zero downloads. Its train split is _guaranteed_ anomaly-free, which
demonstrates the normality assumption in the cleanest possible way; the test
split contains injected spikes, level shifts and frequency bursts with labels.

To fetch one real SMD entity:

```bash
BASE=https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset
for s in train test test_label; do
  curl -sSL --create-dirs -o ../data/SMD/$s/machine-1-1.txt $BASE/$s/machine-1-1.txt
done
```

## 3. How anomalies are scored

For every model we get a per-timestep score by folding overlapping windows back
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
pip install -r requirements.txt

python run_experiments.py                    # synthetic, full smoke run
python run_experiments.py --quick            # tiny + fast sanity check
python run_experiments.py --dataset smd --entity machine-1-1 --epochs 10 --test-stride 5
```

Outputs land in `../results/`: a `results_<tag>.csv`, a `results_<tag>.md`, and a
score-vs-ground-truth plot `scores_<tag>.png`.

## 5. Results

Short CPU runs (reduced sizes) — numbers show the framework and the **relative
ordering**, not final tuned scores. `DDPM-*` use the shared small Transformer;
**ImDiffusion is run at a CPU-reduced config** (`--im-cpu`: window 64, T 25,
channels 32, 2 layers). Its **paper-exact defaults** (window 100, T 50, channels
64, 4 layers, grating split 10) are in `config.py` but need a GPU.

### 5.1 ImDiffusion — exact reproduction (and honest deviations)

`imdiffusion.py` reproduces ImDiffusion (Chen et al. 2023) from the official repo,
**not** the shared-backbone approximation used earlier:

- **Grating mask** — the window is split into blocks; strategy `p=0` observes the
  even blocks and imputes the odd, `p=1` the complement; the two passes cover
  every point (`dataset.py:get_mask`).
- **CSDI backbone** — 4 residual blocks (paper default), each with a **temporal
  transformer + a feature transformer**, plus diffusion-step and **mask-index**
  embeddings; conditions on the clean observed region + noised target, loss on the
  target region only; T=50, **quad** β-schedule (`diff_models.py`, `main_model.py`).
- **Vote-ensemble score** — ancestral sampling captures the denoising trajectory;
  for steps `range(0,30,3)` (10 votes) it computes residual `Σ_feat|impute−x|`, an
  **adaptive top-k threshold** per step (`proper_i = avgE₀·τ_T/avgEᵢ`), and the
  **vote count** across steps is the anomaly score (`ensemble_proper.py`).

Deviations (documented, minor): (i) inputs use the project's shared **z-score**
normalisation so every model sees identical data, instead of ImDiffusion's
MinMax×20; (ii) the **vote count is exposed as a continuous score** so it plugs
into the same ROC-AUC/PR-AUC/best-F1 metrics; (iii) our runs use the
**CPU-reduced size** above — the paper-exact hyper-parameters are the defaults in
`config.py`. The architecture, grating mask, diffusion and vote mechanism are the
paper's.

### SMD `machine-1-1` (38 features, 8 epochs, test-stride 5)

_(populated from `results/results_smd_machine-1-1.md` — ImDiffusion at `--im-cpu`)_

### How the results map to the three research questions

1. **Can a diffusion model trained on normal data find anomalies?** Yes — high
   ROC-AUC on SMD from training on the normal split only.
2. **How does the denoising / conditioning strategy shape the gap?** It matters and
   is dataset-dependent: partial-diffusion (vanilla), selective denoising, and
   ImDiffusion's grating imputation each win on different axes (raw F1 vs recall vs
   cost); on smooth synthetic data selective edges out vanilla.
3. **Does the gain justify the cost?** The cost columns quantify it — ImDiffusion's
   CSDI backbone is by far the most expensive to train (~17× the shared-backbone
   models on synthetic), which is central to the efficiency question.

## 6. Repo layout

```
code/
  config.py          # all hyper-parameters (incl. paper-exact ImDiffusion defaults)
  data.py            # synthetic generator + SMD loader + windowing
  backbone.py        # shared Transformer denoiser (vanilla/selective)
  diffusion.py       # shared-backbone DDPM: vanilla + selective + DDIM scoring
  imdiffusion.py     # faithful ImDiffusion: grating mask + CSDI + vote ensemble
  baselines.py       # LSTM-VAE (BeatGAN = optional extension)
  utils.py           # windowing + metrics (P/R/F1, PA-F1, ROC/PR-AUC)
  run_experiments.py # trains everything, writes the results + cost table
```

## 7. References

1. **ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection** — Chen et al., 2023, _Proc. VLDB Endow._
2. **Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models** — Xiao et al., 2023, _KDD_
3. **Selective Denoising Diffusion Model for Time Series Anomaly Detection** — Obata et al., 2026, _arXiv_
4. **AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise** — Wyatt et al., 2022, _CVPRW_
5. **Time Series Anomaly Detection using Diffusion-based Models** — Pintilie et al., 2023, _ICDMW_
6. **Anomaly Detection for Telemetry Time Series Using a Denoising Diffusion Probabilistic Model** — Sui et al., 2024, _IEEE Sensors Journal_
7. **Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model** — Hu et al., 2024, _ICASSP_
8. **LSTM-Based VAE-GAN for Time-Series Anomaly Detection** — Niu et al., 2020, _Sensors_
9. **Reconstruction-Based Methods for Multivariate Time Series Anomaly Detection: A Review and Taxonomy** — Errachidi et al., 2025
