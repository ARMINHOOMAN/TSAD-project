"""Windowing, metrics and small helpers shared across the project."""
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def make_windows(x: np.ndarray, L: int, stride: int):
    """(T, D) -> (N, L, D) plus the start index of each window."""
    T = x.shape[0]
    starts = list(range(0, max(1, T - L + 1), stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    windows = np.stack([x[s:s + L] for s in starts], axis=0)
    return windows.astype(np.float32), np.array(starts)


def windows_to_series(win_scores: np.ndarray, starts: np.ndarray, T: int, L: int):
    """Fold per-window, per-timestep scores back onto the original series by
    averaging every window that covers a given timestep."""
    acc = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for i, s in enumerate(starts):
        acc[s:s + L] += win_scores[i]
        cnt[s:s + L] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def point_adjust(pred: np.ndarray, label: np.ndarray):
    """Standard point-adjustment: if any point inside a true anomaly segment is
    flagged, the whole segment counts as detected. Inflates scores, so we report
    it alongside the raw numbers rather than instead of them."""
    pred = pred.copy()
    anom = label > 0.5
    i = 0
    n = len(label)
    while i < n:
        if anom[i]:
            j = i
            while j < n and anom[j]:
                j += 1
            if pred[i:j].any():
                pred[i:j] = 1
            i = j
        else:
            i += 1
    return pred


def _candidate_thresholds(score, n_max=400):
    """Thresholds drawn from the score's own quantiles so the search covers the
    tails -- a plain linspace(min, max) misses the sweet spot on heavy-tailed
    scores and collapses to the trivial 'flag-everything' point."""
    qs = np.quantile(score, np.linspace(0.0, 1.0, n_max))
    return np.unique(qs)


def _prf1(pred, label):
    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return f1, prec, rec


def _best_f1_from_preds(score, label, adjust=False):
    best = (0.0, 0.0, 0.0, float(score.min()))
    for thr in _candidate_thresholds(score):
        pred = (score >= thr).astype(int)
        if adjust:
            pred = point_adjust(pred, label)
        f1, prec, rec = _prf1(pred, label)
        if f1 > best[0]:
            best = (f1, prec, rec, float(thr))
    return best


def evaluate_scores(score: np.ndarray, label: np.ndarray):
    """Threshold-free AUCs plus best-F1 (raw and point-adjusted)."""
    label = label.astype(int)
    out = {}
    # AUCs are undefined if only one class is present
    if label.min() != label.max():
        out["roc_auc"] = float(roc_auc_score(label, score))
        out["pr_auc"] = float(average_precision_score(label, score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    f1, p, r, thr = _best_f1_from_preds(score, label, adjust=False)
    out.update(f1=f1, precision=p, recall=r, threshold=float(thr))

    f1a, pa, ra, _ = _best_f1_from_preds(score, label, adjust=True)
    out.update(f1_pa=f1a, precision_pa=pa, recall_pa=ra)
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
