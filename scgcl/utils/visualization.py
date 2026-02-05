"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from sklearn.manifold import TSNE
import warnings


def plot_umap(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Visualization",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 10,
    alpha: float = 0.7,
    cmap: str = 'tab20',
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Create UMAP/t-SNE visualization of embeddings."""
    if X.shape[1] > 2:
        try:
            import umap
            X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        except ImportError:
            warnings.warn("umap-learn not installed, using t-SNE")
            X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        X_2d = X

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]],
                       label=f"Cluster {label}", s=point_size, alpha=alpha)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=point_size, alpha=alpha)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_clusters(
    X: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Side-by-side comparison of true vs predicted clusters."""
    if X.shape[1] > 2:
        try:
            import umap
            X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        except ImportError:
            X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        X_2d = X

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_umap(X_2d, labels_true, title="True Labels", ax=axes[0])
    plot_umap(X_2d, labels_pred, title="Predicted Clusters", ax=axes[1])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    losses: List[float],
    metrics: Optional[dict] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training loss and metrics."""
    n_plots = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    axes = [axes] if n_plots == 1 else axes

    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if metrics:
        for name, values in metrics.items():
            axes[1].plot(values, label=name, linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Evaluation Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
