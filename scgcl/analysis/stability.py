"""Cluster stability analysis using bootstrap resampling."""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Container for cluster stability analysis results."""

    # Overall stability
    mean_ari: float
    std_ari: float

    # Per-cluster stability
    cluster_stability: Dict[int, float] = field(default_factory=dict)

    # Per-cell stability (how often assigned to same cluster)
    cell_stability: np.ndarray = field(default_factory=lambda: np.array([]))

    # Bootstrap label matrix (n_bootstrap x n_cells)
    bootstrap_labels: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    n_bootstrap: int = 0
    n_clusters: int = 0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Cluster Stability Analysis",
            "=" * 50,
            f"Bootstrap iterations: {self.n_bootstrap}",
            f"Number of clusters: {self.n_clusters}",
            f"Overall ARI: {self.mean_ari:.4f} ± {self.std_ari:.4f}",
            "",
            "Per-cluster stability:"
        ]
        for cluster, stability in sorted(self.cluster_stability.items()):
            lines.append(f"  Cluster {cluster}: {stability:.4f}")

        lines.extend([
            "",
            f"Cell stability - Mean: {self.cell_stability.mean():.4f}, "
            f"Min: {self.cell_stability.min():.4f}, Max: {self.cell_stability.max():.4f}",
            "=" * 50
        ])
        return "\n".join(lines)


def cluster_stability(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    clustering_func: Optional[Callable] = None,
    n_clusters: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True
) -> StabilityResult:
    """
    Assess cluster stability using bootstrap resampling.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings (n_cells x n_dims)
    labels : np.ndarray
        Original cluster labels
    n_bootstrap : int
        Number of bootstrap iterations
    sample_fraction : float
        Fraction of cells to sample in each iteration
    clustering_func : callable, optional
        Custom clustering function f(X) -> labels. Uses KMeans if None.
    n_clusters : int, optional
        Number of clusters (inferred from labels if None)
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    StabilityResult
        Stability analysis results
    """
    np.random.seed(seed)

    n_cells = len(labels)
    n_sample = int(n_cells * sample_fraction)

    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    # Default clustering function
    if clustering_func is None:
        def clustering_func(X):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            return kmeans.fit_predict(X)

    # Bootstrap storage
    bootstrap_labels = np.zeros((n_bootstrap, n_cells), dtype=int)
    ari_scores = []

    if verbose:
        logger.info("Running %d bootstrap iterations...", n_bootstrap)

    for i in range(n_bootstrap):
        # Sample cells
        idx = np.random.choice(n_cells, n_sample, replace=False)

        # Cluster subsample
        sub_labels = clustering_func(embeddings[idx])

        # Map back to full dataset using nearest neighbor
        full_labels = np.zeros(n_cells, dtype=int)
        full_labels[idx] = sub_labels

        # For non-sampled cells, assign based on nearest sampled cell
        non_sampled = np.setdiff1d(np.arange(n_cells), idx)
        if len(non_sampled) > 0:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(embeddings[idx])
            _, nearest_idx = nn.kneighbors(embeddings[non_sampled])
            full_labels[non_sampled] = sub_labels[nearest_idx.flatten()]

        bootstrap_labels[i] = full_labels

        # Compute ARI with original labels
        ari = adjusted_rand_score(labels, full_labels)
        ari_scores.append(ari)

        if verbose and (i + 1) % 20 == 0:
            logger.info("  Iteration %d/%d, ARI: %.4f", i + 1, n_bootstrap, ari)

    # Compute per-cell stability
    cell_stability = np.zeros(n_cells)
    for i in range(n_cells):
        cell_labels = bootstrap_labels[:, i]
        # Stability = fraction of times assigned to most common cluster
        unique, counts = np.unique(cell_labels, return_counts=True)
        cell_stability[i] = counts.max() / n_bootstrap

    # Compute per-cluster stability
    cluster_stability_dict = {}
    for cluster in np.unique(labels):
        mask = labels == cluster
        cluster_stability_dict[int(cluster)] = cell_stability[mask].mean()

    result = StabilityResult(
        mean_ari=np.mean(ari_scores),
        std_ari=np.std(ari_scores),
        cluster_stability=cluster_stability_dict,
        cell_stability=cell_stability,
        bootstrap_labels=bootstrap_labels,
        n_bootstrap=n_bootstrap,
        n_clusters=n_clusters
    )

    if verbose:
        logger.info(result.summary())

    return result


def consensus_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    n_iterations: int = 100,
    sample_fraction: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consensus clustering using multiple runs.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings
    n_clusters : int
        Number of clusters
    n_iterations : int
        Number of clustering iterations
    sample_fraction : float
        Fraction of cells per iteration
    seed : int
        Random seed

    Returns
    -------
    labels : np.ndarray
        Consensus cluster labels
    consensus_matrix : np.ndarray
        Co-clustering frequency matrix (n_cells x n_cells)
    """
    np.random.seed(seed)

    n_cells = embeddings.shape[0]
    n_sample = int(n_cells * sample_fraction)

    # Co-clustering matrix
    coassoc = np.zeros((n_cells, n_cells))
    counts = np.zeros((n_cells, n_cells))

    for i in range(n_iterations):
        idx = np.random.choice(n_cells, n_sample, replace=False)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + i)
        sub_labels = kmeans.fit_predict(embeddings[idx])

        # Update co-clustering matrix
        for c in range(n_clusters):
            cluster_idx = idx[sub_labels == c]
            for j in cluster_idx:
                for k in cluster_idx:
                    coassoc[j, k] += 1

        # Update counts
        for j in idx:
            for k in idx:
                counts[j, k] += 1

    # Normalize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        consensus_matrix = np.divide(coassoc, counts, where=counts > 0)
        consensus_matrix = np.nan_to_num(consensus_matrix)

    # Final clustering on consensus matrix
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    # Convert similarity to distance
    distance_matrix = 1 - consensus_matrix
    np.fill_diagonal(distance_matrix, 0)
    labels = clustering.fit_predict(distance_matrix)

    return labels, consensus_matrix


def plot_stability(
    result: StabilityResult,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot cluster stability results.

    Parameters
    ----------
    result : StabilityResult
        Stability analysis results
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Per-cluster stability
    clusters = sorted(result.cluster_stability.keys())
    stabilities = [result.cluster_stability[c] for c in clusters]
    axes[0].bar(clusters, stabilities, color='steelblue')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Stability')
    axes[0].set_title('Per-Cluster Stability')
    axes[0].set_ylim(0, 1)

    # Cell stability distribution
    axes[1].hist(result.cell_stability, bins=30, color='steelblue', edgecolor='white')
    axes[1].axvline(result.cell_stability.mean(), color='red', linestyle='--',
                    label=f'Mean: {result.cell_stability.mean():.3f}')
    axes[1].set_xlabel('Stability Score')
    axes[1].set_ylabel('Number of Cells')
    axes[1].set_title('Cell Stability Distribution')
    axes[1].legend()

    # ARI distribution across bootstraps
    # Recompute ARIs from bootstrap labels
    from sklearn.metrics import adjusted_rand_score
    # We need original labels - approximate from mode
    mode_labels = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 0, result.bootstrap_labels
    )
    ari_scores = [
        adjusted_rand_score(mode_labels, result.bootstrap_labels[i])
        for i in range(result.n_bootstrap)
    ]

    axes[2].hist(ari_scores, bins=20, color='steelblue', edgecolor='white')
    axes[2].axvline(np.mean(ari_scores), color='red', linestyle='--',
                    label=f'Mean: {np.mean(ari_scores):.3f}')
    axes[2].set_xlabel('ARI')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Bootstrap ARI Distribution')
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
