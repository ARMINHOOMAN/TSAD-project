#  Multivariate Time-Series Anomaly Detection

. Four unsupervised detectors are trained on the normal split of the **Server
Machine Dataset (SMD)** and compared on the labelled test split We aim to show how well a generative model
captures normal behaviour, and how the resulting anomaly score is turned into a
decision.

---

## Table of contents

1. [Models](#1-models)
2. [Dataset](#2-dataset)
3. [Scoring and evaluation](#3-scoring-and-evaluation)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Results](#6-results)
7. [Repository layout](#7-repository-layout)
8. [Module dependencies](#8-module-dependencies)
9. [References](#9-references)

---

## 1. Models

| Model | Family | Noise design / idea | Denoiser | Follows |
|---|---|---|---|---|
| **LSTM-VAE** | encoder–decoder baseline | reconstruct the window from a KL-regularised latent vector | LSTM encoder–decoder | [6] |
| **DDPM-vanilla** | diffusion baseline | full Gaussian noising; partial-diffusion reconstruction | small Transformer | [3], [7] |
| **DDPM-selective** | selective denoising | mask the *noise* during training with a Bernoulli mask; denoise the raw instance with **no noise added** at inference | **CSDI** | [2] |
| **ImDiffusion** | imputation-based diffusion | grating mask hides part of the window; the rest conditions the model; step-wise **vote ensemble** over the reverse trajectory | **CSDI** | [1] |

**Backbones.** `DDPM-selective` and `ImDiffusion` are built on the *same* CSDI
architecture [4], which includes four residual blocks, each combining a temporal Transformer
layer and a feature Transformer layer. `anomalyfilter.py` imports `DiffCSDI`
directly from `imdiffusion.py`, so the two methods are architecturally identical
and differ only in their noise design and decision rule, which is what makes the
comparison between them informative. `DDPM-vanilla` uses a **simpler Transformer
denoiser** (`backbone.py`) and serves as a reference point for what a plain
diffusion model achieves.

---

## 2. Dataset

**Server Machine Dataset (SMD)** : Five weeks of 38-dimensional server telemetry
from a large internet company (OmniAnomaly release). The experiments use dataset
`machine-1-1`.

The train/test split of SMD are as below:

- the **train** split is a curated normal-operation period with no labelled
  anomalies, and is the only data the models ever see during training;
- the **test** split carries point-level labels: 28,479 timesteps of which 2,694
  (9.46 %) are anomalous, grouped into eight segments;
- z-score normalisation statistics are computed on the **train** split only.


Download one entity:

```bash
BASE=https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset
for s in train test test_label; do
  curl -sSL --create-dirs -o data/SMD/$s/machine-1-1.txt $BASE/$s/machine-1-1.txt
done
```

---

## 3. Scoring and evaluation

Every model produces one anomaly score per timestep by folding windowed scores
back onto the series:

- **LSTM-VAE, DDPM-vanilla, DDPM-selective**: They calculate the mean squared reconstruction error
  between the input window and its reconstruction. 
- **ImDiffusion** — a **vote count**: At each retained denoising step a residual
  is thresholded at an adaptive percentile `tau_t = (sum E_T / sum E_t) * tau_T`,
  and the votes are summed over ten steps. The score is therefore an integer in
  `0..10` rather than a continuous value.

Reported metrics:

| Metric | Note |
|---|---|
| best F1, precision, recall |threshold chosen by sweeping the score's own quantiles |
| point-adjusted F1 (`f1_pa`) |  a whole segment counts as detected if one point in it is flagged |
| ROC-AUC, PR-AUC | the only metrics free of threshold selection |


---

## 4. Installation

```bash
git clone https://github.com/ARMINHOOMAN/Anomaly-dtetection-diffusion-model.git
cd Anomaly-dtetection-diffusion-model
pip install -r code/requirements.txt
```

Requires Python 3.10+ and PyTorch. A CUDA GPU is used automatically when
available; the reported runs used a single NVIDIA RTX 5070.

---

## 5. Usage

All commands are run **from inside `code/`**, because the default data and output
paths are relative to that directory.

```bash
cd code

# full run: SMD machine-1-1, all four models (this reproduces the reported table)
python run_experiments.py --out ../results

# fast sanity check with tiny models
python run_experiments.py --quick

```

### Outputs

Written to `--out`:

```
results_smd_machine-1-1.csv      metrics table
results_smd_machine-1-1.md       the same table in Markdown
roc_smd_machine-1-1.png          ROC curves, all models + random baseline
pr_smd_machine-1-1.png           precision-recall curves, all models + random baseline
imdiff_score_smd_machine-1-1.png ImDiffusion imputation vs ground truth, and its vote score
imdiff_ensemble_smd_machine-1-1.png  per-step thresholds and the final vote
scores_smd_machine-1-1.png       anomaly score of the best model vs ground truth
```

If a target file is locked (for example open in Excel), the writer falls back to a
timestamped filename instead of losing the run.

---

## 6. Results

SMD `machine-1-1`, 38 features, 50 epochs. The random detector is included as a
reference; its precision is pinned at the anomaly rate and its point-adjusted F1
is almost saturated. Best value among the four models in **bold**.

| Model | F1 | Precision | Recall | F1-PA | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LSTM-VAE | 0.607 | 0.589 | 0.625 | 0.998 | 0.896 | 0.567 |
| DDPM-vanilla | 0.602 | 0.548 | 0.667 | **0.999** | 0.932 | 0.614 |
| DDPM-selective | **0.658** | 0.547 | **0.826** | 0.998 | **0.957** | **0.691** |
| ImDiffusion | 0.322 | **0.829** | 0.200 | 0.988 | 0.598 | 0.248 |
| *Random detector* | *0.173* | *0.095* | *0.985* | *0.968* | *0.500* | *0.095* |


---

## 7. Repository layout

```
.
├── code/
│   ├── run_experiments.py   entry point: trains all models, evaluates, writes tables and figures
│   ├── config.py            all hyperparameters, one dataclass per model
│   ├── data.py              SMD loader, synthetic generator, z-score normalisation, windowing
│   ├── utils.py             windowing, folding, metrics (P/R/F1, PA-F1, ROC-AUC, PR-AUC), seeding
│   ├── backbone.py          simple Transformer denoiser used by DDPM-vanilla
│   ├── diffusion.py         DDPM-vanilla: forward process, DDIM sampling, reconstruction score
│   ├── baselines.py         LSTM-VAE
│   ├── imdiffusion.py       ImDiffusion: CSDI backbone (DiffCSDI), grating mask, vote ensemble
│   ├── anomalyfilter.py     DDPM-selective: masked Gaussian noise, noiseless inference
│   ├── plots.py             ROC/PR curves and the ImDiffusion diagnostic figures
│   ├── requirements.txt
│   └── README.md            implementation notes and deviations from the original papers
├── data/SMD/                train/, test/, test_label/ (downloaded separately, git-ignored)
├── results/                 metrics tables and figures
├── report/                  final report (LaTeX source, bibliography, figures)
├── RUN.bat                  Windows launcher
└── README.md
```

---

## 8. Module dependencies


| Module | Imports from the project | Provides | Consumed by |
|---|---|---|---|
| `config.py` | — | `Config` and the per-model dataclasses | every model, `run_experiments.py` |
| `utils.py` | — | `make_windows`, `windows_to_series`, `evaluate_scores`, `point_adjust`, `count_params`, `set_seed` | `data.py`, `imdiffusion.py`, `anomalyfilter.py`, `plots.py`, `run_experiments.py` |
| `backbone.py` | — | `Denoiser` (simple Transformer) | `diffusion.py` |
| `baselines.py` | — | `LSTMVAE` | `run_experiments.py` |
| `data.py` | `utils` | `get_data`, `WindowDataset` | `run_experiments.py` |
| `diffusion.py` | `backbone` | `build_diffusion` → `GaussianDiffusion` | `run_experiments.py` |
| `imdiffusion.py` | `utils` | `DiffCSDI`, `ImDiffusion`, `train_imdiffusion`, `score_imdiffusion` | `run_experiments.py`, `anomalyfilter.py` |
| `anomalyfilter.py` | `imdiffusion`, `utils` | `AnomalyFilter`, `train_anomalyfilter`, `score_anomalyfilter` | `run_experiments.py` |
| `plots.py` | `utils` | `plot_roc`, `plot_pr`, `plot_imdiff_score`, `plot_imdiff_ensemble` | `run_experiments.py` |

**Prerequisites and data flow.** `config.py` and `utils.py` are leaf modules with
no internal dependencies and must be importable before anything else. The one
cross-model dependency worth knowing is that **`anomalyfilter.py` imports
`DiffCSDI` from `imdiffusion.py`** — this is deliberate, and it is what guarantees
that DDPM-selective and ImDiffusion share an identical backbone. Editing
`imdiffusion.py` therefore changes both models.


---

## 9. References

[1] Y. Chen, C. Zhang, M. Ma, Y. Liu, R. Ding, B. Li, S. He, S. Rajmohan, Q. Lin,
and D. Zhang. "ImDiffusion: Imputed Diffusion Models for Multivariate Time Series
Anomaly Detection." *Proceedings of the VLDB Endowment*, 17(3):359–372, 2023.

[2] K. Obata, Z. Chen, Y. Matsubara, L. Zhu, and Y. Sakurai. "Selective Denoising
Diffusion Model for Time Series Anomaly Detection." *arXiv preprint
arXiv:2602.23662*, 2026.

[3] J. Ho, A. Jain, and P. Abbeel. "Denoising Diffusion Probabilistic Models."
*Advances in Neural Information Processing Systems*, 33:6840–6851, 2020.

[4] Y. Tashiro, J. Song, Y. Song, and S. Ermon. "CSDI: Conditional Score-based
Diffusion Models for Probabilistic Time Series Imputation." *Advances in Neural
Information Processing Systems*, 34:24804–24816, 2021.

[5] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei. "Robust Anomaly Detection
for Multivariate Time Series through Stochastic Recurrent Neural Network." In
*Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining*, pages 2828–2837, 2019.

[6] D. Park, Y. Hoshi, and C. C. Kemp. "A Multimodal Anomaly Detector for
Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder." *IEEE
Robotics and Automation Letters*, 3(3):1544–1551, 2018.

[7] J. Wyatt, A. Leach, S. M. Schmon, and C. G. Willcocks. "AnoDDPM: Anomaly
Detection with Denoising Diffusion Probabilistic Models using Simplex Noise." In
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
Workshops (CVPRW)*, pages 650–656, 2022.

[8] J. Xu, H. Wu, J. Wang, and M. Long. "Anomaly Transformer: Time Series Anomaly
Detection with Association Discrepancy." In *International Conference on Learning
Representations*, 2022.

[9] S. Tuli, G. Casale, and N. R. Jennings. "TranAD: Deep Transformer Networks for
Anomaly Detection in Multivariate Time Series Data." *Proceedings of the VLDB
Endowment*, 15(6):1201–1214, 2022.

[10] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon. "Towards a Rigorous
Evaluation of Time-Series Anomaly Detection." In *Proceedings of the AAAI
Conference on Artificial Intelligence*, volume 36, pages 7194–7201, 2022.

[11] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, and M. J.
Franklin. "Volume Under the Surface: A New Accuracy Evaluation Measure for
Time-Series Anomaly Detection." *Proceedings of the VLDB Endowment*,
15(11):2774–2787, 2022.

[12] C. Xiao, Z. Gou, W. Tai, K. Zhang, and F. Zhou. "Imputation-based Time-Series
Anomaly Detection with Conditional Weight-Incremental Diffusion Models." In
*Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining*, pages 2742–2751, 2023.
