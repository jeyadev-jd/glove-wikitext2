"""Plots: training loss curve and PCA projection of learned embeddings.

matplotlib only (no seaborn), with the non-interactive Agg backend so the
pipeline runs headless.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.evaluation import EmbeddingIndex  # noqa: E402


def plot_loss_curve(epochs: Sequence[int], losses: Sequence[float],
                    output_path: str, title: str = "GloVe training loss") -> str:
    """Save an epoch-vs-loss line plot to ``output_path``."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4.5), dpi=150)
    axes.plot(list(epochs), list(losses), marker="o", markersize=3,
              linewidth=1.5, color="#1f4e79")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Mean weighted squared error")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_multiple_loss_curves(series: dict, output_path: str,
                              title: str = "Loss curves") -> str:
    """Overlay several ``{label: (epochs, losses)}`` curves in one figure."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4.5), dpi=150)
    for label, (epochs, losses) in series.items():
        axes.plot(list(epochs), list(losses), linewidth=1.4, label=str(label))
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Mean weighted squared error")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def select_visualization_words(index: EmbeddingIndex, candidates: Sequence[str],
                               count: int = 80) -> List[str]:
    """Pick up to ``count`` in-vocabulary words, topping up from the vocabulary."""
    chosen = [w for w in candidates if w in index]
    for word in index.idx_to_word:
        if len(chosen) >= count:
            break
        if word not in chosen and len(word) > 2:
            chosen.append(word)
    return chosen[:count]


def plot_pca_embeddings(index: EmbeddingIndex, words: Sequence[str],
                        output_path: str, seed: int = 42,
                        title: Optional[str] = None) -> str:
    """Project the given words' vectors to 2D with PCA and save a labelled plot."""
    valid = [w for w in words if w in index]
    if len(valid) < 3:
        raise ValueError("need at least 3 in-vocabulary words for a PCA plot")

    matrix = np.vstack([index.vector(w) for w in valid])
    coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure, axes = plt.subplots(figsize=(11, 8.5), dpi=150)
    axes.scatter(coords[:, 0], coords[:, 1], s=14, color="#1f4e79", alpha=0.75)
    for (x, y), word in zip(coords, valid):
        axes.annotate(word, (x, y), fontsize=7, alpha=0.85,
                      xytext=(3, 3), textcoords="offset points")
    axes.set_xlabel("PC 1")
    axes.set_ylabel("PC 2")
    axes.set_title(title or f"PCA of {len(valid)} learned word vectors")
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_ablation_bars(labels: Sequence[str], values: Sequence[float],
                       output_path: str, ylabel: str, title: str) -> str:
    """Simple bar chart used to summarise ablation outcomes."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4.5), dpi=150)
    axes.bar(range(len(labels)), list(values), color="#1f4e79", alpha=0.85)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(list(labels), rotation=30, ha="right", fontsize=8)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
