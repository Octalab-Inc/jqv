"""Accuracy and calibration metrics + reliability diagram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def accuracy(probs, labels) -> float:
    p, y = _np(probs), _np(labels)
    return float((p.argmax(-1) == y).mean())


def nll(probs, labels, eps: float = 1e-12) -> float:
    p, y = _np(probs), _np(labels)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1)).mean())


def brier(probs, labels) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - y_k)^2. Padded (nan) choices ignored."""
    p, y = _np(probs).copy(), _np(labels)
    p = np.nan_to_num(p, nan=0.0)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(-1).mean())


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected calibration error on the top-1 confidence, equal-width bins."""
    p, y = _np(probs), _np(labels)
    p = np.nan_to_num(p, nan=0.0)
    conf, pred = p.max(-1), p.argmax(-1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def summary(probs, labels, n_bins: int = 10) -> dict[str, float]:
    return {
        "n": int(len(_np(labels))),
        "accuracy": accuracy(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "ece": ece(probs, labels, n_bins),
        "mean_confidence": float(np.nan_to_num(_np(probs), nan=0.0).max(-1).mean()),
    }


def reliability_diagram(probs, labels, path: str | Path, title: str = "", n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p, y = _np(probs), _np(labels)
    p = np.nan_to_num(p, nan=0.0)
    conf, pred = p.max(-1), p.argmax(-1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    accs, counts, confs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        counts.append(int(m.sum()))
        accs.append(correct[m].mean() if m.any() else np.nan)
        confs.append(conf[m].mean() if m.any() else np.nan)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], color="#9A9A9A", lw=1, ls="--", zorder=1)
    ax.bar(centers, np.nan_to_num(accs), width=1 / n_bins * 0.9, color="#3B6FB6", edgecolor="white", lw=1, zorder=2)
    for c, a, n in zip(centers, accs, counts):
        if n:
            ax.text(c, (a if not np.isnan(a) else 0) + 0.02, f"n={n}", ha="center", va="bottom", fontsize=6, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("confidence (top-1 probability)")
    ax.set_ylabel("accuracy")
    e = ece(p, y, n_bins)
    ax.set_title(f"{title}  ECE={e:.3f}".strip(), fontsize=10)
    ax.grid(color="#EEEEEE", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
