"""Figures for the results folder.

  roc_<tag>.png       ROC curves, every model + the random-classifier diagonal
  pr_<tag>.png        Precision-Recall curves, every model + the random baseline
  imdiff_score_<tag>.png    imputation vs ground truth + vote score over the series
  imdiff_ensemble_<tag>.png per-denoising-step residual, threshold and vote
                            (the paper's Fig. 8)

Everything here is read-only w.r.t. the models: it consumes the score arrays and
the diagnostics dict that score_imdiffusion() returns.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)

from utils import windows_to_series

COLORS = {"LSTM-VAE": "#4C72B0", "DDPM-vanilla": "#DD8452",
          "DDPM-selective": "#55A868", "ImDiffusion": "#C44E52"}


def _c(name, i=0):
    return COLORS.get(name, plt.cm.tab10(i % 10))


# --------------------------------------------------------------------------
# 1 + 2.  ROC and PR, all models against the random baseline
# --------------------------------------------------------------------------
def plot_roc(score_curves, labels, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for i, (name, s) in enumerate(score_curves.items()):
        fpr, tpr, _ = roc_curve(labels, s)
        ax.plot(fpr, tpr, lw=1.8, color=_c(name, i),
                label=f"{name}  (AUC = {roc_auc_score(labels, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="random  (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — SMD machine-1-1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(score_curves, labels, path):
    pi = float(np.mean(labels))
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for i, (name, s) in enumerate(score_curves.items()):
        prec, rec, _ = precision_recall_curve(labels, s)
        ax.plot(rec, prec, lw=1.8, color=_c(name, i),
                label=f"{name}  (AP = {average_precision_score(labels, s):.3f})")
    ax.axhline(pi, ls="--", c="k", lw=1.2,
               label=f"random  (AP = {pi:.3f} = anomaly rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — SMD machine-1-1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _segments(labels):
    """[(start, end), ...] for each contiguous anomaly run."""
    seg, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] > 0.5:
            j = i
            while j < n and labels[j] > 0.5:
                j += 1
            seg.append((i, j))
            i = j
        else:
            i += 1
    return seg


def _shade(ax, labels, lo, hi, first_label=True):
    for a, b in _segments(labels):
        if b < lo or a > hi:
            continue
        ax.axvspan(max(a, lo), min(b, hi), color="#e8746e", alpha=.22, lw=0,
                   label="true anomaly" if first_label else None)
        first_label = False
    return first_label


def _pick_window(labels, width=300):
    """A narrow slice straddling the ONSET of an anomaly, so the panel shows
    normal behaviour (white) running into the anomaly (light red).

    Centring on a segment's midpoint is wrong for SMD: its segments run to 721
    steps, so any window narrower than that lands wholly inside the anomaly and
    the figure has no normal baseline to compare against.
    """
    seg = _segments(labels)
    if not seg:
        return 0, min(width, len(labels))
    half = width // 2
    # longest segment that can fill the right half of the window
    a, _ = max([s for s in seg if s[1] - s[0] >= half] or seg,
               key=lambda s: s[1] - s[0])
    lo = max(0, a - half)
    return lo, min(len(labels), lo + width)


def _pick_feature(series, labels):
    """The most legible channel to draw: the one whose values separate anomalous
    from normal periods most strongly. Labels are used for readability of the
    figure only -- nothing in the detector sees this."""
    a = labels > 0.5
    if a.all() or not a.any():
        return int(np.argmax(series.std(axis=0)))
    sep = np.abs(series[a].mean(0) - series[~a].mean(0)) / (series.std(0) + 1e-9)
    return int(np.argmax(sep))


def _fold_feature(imp_s, starts, T_total, L, feat):
    """(n_windows, L, K) imputations -> (T,) series for one feature."""
    return windows_to_series(imp_s[:, :, feat], starts, T_total, L)


# --------------------------------------------------------------------------
# 3.  imputation vs ground truth, and the vote score over the whole test set
# --------------------------------------------------------------------------
def plot_imdiff_score(details, test_series, labels, path, feat=None):
    T_total = len(labels)
    keep, L = details["keep"], details["L"]
    final = min(keep)                                   # t = 0, fully denoised
    imp = details["imp"][final]
    starts = details["starts"]

    if feat is None:
        feat = _pick_feature(test_series, labels)

    x = test_series[:, feat]
    xhat = _fold_feature(imp, starts, T_total, L, feat)
    resid = np.abs(x - xhat)
    votes = details["votes"]
    lo, hi = _pick_window(labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4),
                             gridspec_kw=dict(height_ratios=[1, 1]))

    # series and imputation share the left axis; the error lives on its own
    # right axis -- summed-squared error is orders of magnitude larger and
    # would otherwise flatten both lines onto zero.
    ax = axes[0]
    t = np.arange(lo, hi)
    l1, = ax.plot(t, x[lo:hi], lw=1.0, color="#4C72B0", label="time series")
    l2, = ax.plot(t, xhat[lo:hi], lw=1.0, color="#DD8452", label="imputed series")
    ax2 = ax.twinx()
    l3, = ax2.plot(t, resid[lo:hi], lw=1.0, color="#55A868", alpha=.85,
                   label="imputed error (right)")
    ax2.set_ylabel("|error|", color="#55A868", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#55A868", labelsize=8)
    _shade(ax, labels, lo, hi)
    ax.set_xlim(lo, hi)
    ax.set_title(f"ImDiffusion imputation vs ground truth "
                 f"(feature {feat}, t = {lo}–{hi})")
    ax.set_ylabel("value")
    ax.legend(handles=[l1, l2, l3], ncol=3, fontsize=8, loc="upper left",
              framealpha=.95)

    ax = axes[1]
    ax.plot(votes, lw=.7, color="#C44E52", label="vote count $V_l$")
    _shade(ax, labels, 0, T_total)
    ax.set_xlim(0, T_total)
    ax.set_ylim(-.3, details["n_votes"] + .3)
    ax.set_title("ImDiffusion anomaly score over the full SMD test series")
    ax.set_xlabel("time")
    ax.set_ylabel("votes")
    ax.legend(ncol=2, fontsize=8, loc="upper left", framealpha=.95)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return feat


# --------------------------------------------------------------------------
# 4.  ensemble inference: one panel per denoising step (paper Fig. 8)
# --------------------------------------------------------------------------
def plot_imdiff_ensemble(details, test_series, labels, path, xi=None,
                         feat=0, width=300):
    T_total = len(labels)
    keep, L, T = details["keep"], details["L"], details["T"]
    starts = details["starts"]
    lo, hi = _pick_window(labels, width)
    t = np.arange(lo, hi)

    # All of `keep` votes; the figure shows only the FIRST voting step (noisiest)
    # and the LAST (fully denoised), so the panel stays readable. Reverse order =
    # most-noisy first, matching the paper's "denoising step" numbering.
    ordered = sorted(keep, reverse=True)
    steps = [ordered[0], ordered[-1]] if len(ordered) > 1 else ordered
    n = len(steps)
    fig, axes = plt.subplots(n + 1, 1, figsize=(10, 2.1 * n + 2.4), sharex=True)

    for ax, s in zip(axes[:n], steps):
        xhat = _fold_feature(details["imp"][s], starts, T_total, L, feat)
        r = details["step_resid"][s]
        pred = details["step_pred"][s]
        ax.plot(t, test_series[lo:hi, feat], lw=.8, color="#4C72B0")
        ax.plot(t, xhat[lo:hi], lw=.8, color="#DD8452")
        # residual + its threshold on a separate axis (different magnitude)
        ax2 = ax.twinx()
        ax2.plot(t, r[lo:hi], lw=.8, color="#55A868", alpha=.85)
        ax2.axhline(details["step_thr"][s], ls=":", lw=1.0, color="#8172B3")
        ax2.tick_params(axis="y", labelcolor="#55A868", labelsize=6)
        flag = t[pred[lo:hi] > .5]
        if len(flag):
            ax.plot(flag, np.full(len(flag), ax.get_ylim()[0]), "|",
                    ms=5, color="#937860", clip_on=False)
        _shade(ax, labels, lo, hi)
        ax.set_ylabel(f"denoising\nstep {T - s}", fontsize=7.5, rotation=0,
                      ha="right", va="center")
        ax.tick_params(labelsize=7)

    ax = axes[n]
    v = details["votes"][lo:hi]
    ax.bar(t, v, width=1.0, color="#C44E52", alpha=.85, label="total votes")
    if xi is not None:
        ax.axhline(xi, ls="--", lw=1.2, color="k",
                   label=rf"anomaly threshold $\xi$ = {xi:g}")
    _shade(ax, labels, lo, hi)
    ax.set_ylim(0, details["n_votes"] + .5)
    ax.set_ylabel("votes", fontsize=8)
    ax.set_xlabel("time")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=.95)
    ax.tick_params(labelsize=7)

    handles = [plt.Line2D([], [], color="#4C72B0", lw=1, label="time series"),
               plt.Line2D([], [], color="#DD8452", lw=1, label="imputed series"),
               plt.Line2D([], [], color="#55A868", lw=1,
                          label="imputed error (right axis)"),
               plt.Line2D([], [], color="#8172B3", ls=":", lw=1,
                          label=r"per-step threshold $\tau_t$"),
               plt.Line2D([], [], color="#937860", marker="|", ls="none",
                          label="step prediction $Y_t$")]
    fig.suptitle("ImDiffusion ensemble inference: first and last voting step "
                 f"of {details['n_votes']} (feature {feat})", fontsize=11, y=.995)
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(.5, .965))
    top = 1.0 - (0.72 / fig.get_figheight())     # room for title + legend
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(path, dpi=150)
    plt.close(fig)
