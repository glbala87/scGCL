"""Quality control metrics and batch visualization."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterQCResult:
    """Container for cluster QC results."""

    cluster_stats: pd.DataFrame
    cell_stats: pd.DataFrame
    overall_stats: Dict

    def summary(self) -> Dict:
        """Get summary statistics."""
        return self.overall_stats


def compute_cluster_qc(
    expression: np.ndarray,
    labels: np.ndarray,
    gene_names: Optional[np.ndarray] = None,
    mt_prefix: str = 'MT-',
    rb_prefix: str = 'RP'
) -> ClusterQCResult:
    """
    Compute QC metrics per cluster.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray
        Cluster labels
    gene_names : np.ndarray, optional
        Gene names for MT/RB calculation
    mt_prefix : str
        Prefix for mitochondrial genes
    rb_prefix : str
        Prefix for ribosomal genes

    Returns
    -------
    ClusterQCResult
        QC metrics per cluster
    """
    n_cells, n_genes = expression.shape
    unique_clusters = sorted(np.unique(labels))

    # Per-cell metrics
    cell_stats = pd.DataFrame()
    cell_stats['cluster'] = labels

    # Total counts per cell
    total_counts = np.sum(expression, axis=1)
    cell_stats['total_counts'] = total_counts

    # Number of genes detected
    n_genes_detected = np.sum(expression > 0, axis=1)
    cell_stats['n_genes'] = n_genes_detected

    # MT and RB percentages if gene names provided
    if gene_names is not None:
        gene_names_upper = np.array([g.upper() for g in gene_names])

        # Mitochondrial
        mt_mask = np.char.startswith(gene_names_upper, mt_prefix.upper())
        if np.any(mt_mask):
            mt_counts = np.sum(expression[:, mt_mask], axis=1)
            cell_stats['pct_mt'] = 100 * mt_counts / np.maximum(total_counts, 1)
        else:
            cell_stats['pct_mt'] = 0.0

        # Ribosomal
        rb_mask = np.char.startswith(gene_names_upper, rb_prefix.upper())
        if np.any(rb_mask):
            rb_counts = np.sum(expression[:, rb_mask], axis=1)
            cell_stats['pct_rb'] = 100 * rb_counts / np.maximum(total_counts, 1)
        else:
            cell_stats['pct_rb'] = 0.0

    # Per-cluster metrics
    cluster_data = []

    for cluster in unique_clusters:
        mask = labels == cluster
        n_cluster = np.sum(mask)

        cluster_expr = expression[mask]
        cluster_total = total_counts[mask]
        cluster_ngenes = n_genes_detected[mask]

        stats_dict = {
            'cluster': cluster,
            'n_cells': n_cluster,
            'mean_counts': np.mean(cluster_total),
            'median_counts': np.median(cluster_total),
            'std_counts': np.std(cluster_total),
            'mean_n_genes': np.mean(cluster_ngenes),
            'median_n_genes': np.median(cluster_ngenes),
            'std_n_genes': np.std(cluster_ngenes)
        }

        if 'pct_mt' in cell_stats.columns:
            mt_values = cell_stats.loc[mask, 'pct_mt'].values
            stats_dict['mean_pct_mt'] = np.mean(mt_values)
            stats_dict['median_pct_mt'] = np.median(mt_values)

        if 'pct_rb' in cell_stats.columns:
            rb_values = cell_stats.loc[mask, 'pct_rb'].values
            stats_dict['mean_pct_rb'] = np.mean(rb_values)
            stats_dict['median_pct_rb'] = np.median(rb_values)

        cluster_data.append(stats_dict)

    cluster_stats = pd.DataFrame(cluster_data)

    # Overall statistics
    overall_stats = {
        'n_cells': n_cells,
        'n_clusters': len(unique_clusters),
        'mean_cluster_size': n_cells / len(unique_clusters),
        'min_cluster_size': int(cluster_stats['n_cells'].min()),
        'max_cluster_size': int(cluster_stats['n_cells'].max()),
        'mean_total_counts': float(np.mean(total_counts)),
        'mean_n_genes': float(np.mean(n_genes_detected))
    }

    return ClusterQCResult(
        cluster_stats=cluster_stats,
        cell_stats=cell_stats,
        overall_stats=overall_stats
    )


def compute_cluster_purity(
    labels: np.ndarray,
    true_labels: np.ndarray
) -> pd.DataFrame:
    """
    Compute cluster purity given true labels.

    Parameters
    ----------
    labels : np.ndarray
        Predicted cluster labels
    true_labels : np.ndarray
        Ground truth labels

    Returns
    -------
    pd.DataFrame
        Purity metrics per cluster
    """
    unique_clusters = sorted(np.unique(labels))

    results = []
    for cluster in unique_clusters:
        mask = labels == cluster
        cluster_true = true_labels[mask]

        # Get most common true label
        values, counts = np.unique(cluster_true, return_counts=True)
        max_idx = np.argmax(counts)

        purity = counts[max_idx] / len(cluster_true)
        majority_label = values[max_idx]

        results.append({
            'cluster': cluster,
            'n_cells': np.sum(mask),
            'majority_label': majority_label,
            'majority_count': counts[max_idx],
            'purity': purity,
            'n_labels': len(values)
        })

    return pd.DataFrame(results)


def compute_batch_mixing(
    embedding: np.ndarray,
    batch: np.ndarray,
    n_neighbors: int = 50
) -> Dict:
    """
    Compute batch mixing metrics.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates (cells x dims)
    batch : np.ndarray
        Batch labels
    n_neighbors : int
        Number of neighbors

    Returns
    -------
    Dict
        Batch mixing metrics
    """
    from scipy.spatial.distance import cdist

    n_cells = embedding.shape[0]
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Compute distances
    distances = cdist(embedding, embedding, metric='euclidean')

    # For each cell, compute batch mixing in neighborhood
    mixing_scores = []

    for i in range(n_cells):
        # Get k nearest neighbors (excluding self)
        dists = distances[i].copy()
        dists[i] = np.inf
        neighbor_idx = np.argsort(dists)[:n_neighbors]

        # Compute batch entropy in neighborhood
        neighbor_batches = batch[neighbor_idx]
        _, counts = np.unique(neighbor_batches, return_counts=True)
        props = counts / n_neighbors

        # Entropy (normalized by max entropy)
        entropy = -np.sum(props * np.log(props + 1e-10))
        max_entropy = np.log(n_batches)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        mixing_scores.append(normalized_entropy)

    mixing_scores = np.array(mixing_scores)

    # Per-batch statistics
    batch_stats = {}
    for b in unique_batches:
        mask = batch == b
        batch_stats[str(b)] = {
            'n_cells': int(np.sum(mask)),
            'mean_mixing': float(np.mean(mixing_scores[mask])),
            'median_mixing': float(np.median(mixing_scores[mask]))
        }

    return {
        'overall_mixing': float(np.mean(mixing_scores)),
        'mixing_scores': mixing_scores,
        'n_batches': n_batches,
        'batch_stats': batch_stats
    }


def plot_cluster_qc(
    qc_result: ClusterQCResult,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot cluster QC metrics.

    Parameters
    ----------
    qc_result : ClusterQCResult
        Results from compute_cluster_qc
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    cluster_stats = qc_result.cluster_stats
    cell_stats = qc_result.cell_stats

    n_clusters = len(cluster_stats)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Cluster sizes
    ax = axes[0, 0]
    ax.bar(cluster_stats['cluster'], cluster_stats['n_cells'])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of cells')
    ax.set_title('Cluster Sizes')

    # Total counts per cluster (boxplot)
    ax = axes[0, 1]
    data = [cell_stats.loc[cell_stats['cluster'] == c, 'total_counts'].values
            for c in sorted(cell_stats['cluster'].unique())]
    ax.boxplot(data, labels=sorted(cell_stats['cluster'].unique()))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Total counts')
    ax.set_title('Total Counts per Cluster')

    # N genes per cluster (boxplot)
    ax = axes[0, 2]
    data = [cell_stats.loc[cell_stats['cluster'] == c, 'n_genes'].values
            for c in sorted(cell_stats['cluster'].unique())]
    ax.boxplot(data, labels=sorted(cell_stats['cluster'].unique()))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Genes detected')
    ax.set_title('Genes Detected per Cluster')

    # MT percentage if available
    ax = axes[1, 0]
    if 'pct_mt' in cell_stats.columns and cell_stats['pct_mt'].sum() > 0:
        data = [cell_stats.loc[cell_stats['cluster'] == c, 'pct_mt'].values
                for c in sorted(cell_stats['cluster'].unique())]
        ax.boxplot(data, labels=sorted(cell_stats['cluster'].unique()))
        ax.set_xlabel('Cluster')
        ax.set_ylabel('% MT')
        ax.set_title('Mitochondrial % per Cluster')
    else:
        ax.text(0.5, 0.5, 'MT data not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Mitochondrial %')

    # Counts vs genes scatter
    ax = axes[1, 1]
    scatter = ax.scatter(
        cell_stats['total_counts'],
        cell_stats['n_genes'],
        c=cell_stats['cluster'],
        cmap='tab20',
        alpha=0.5,
        s=5
    )
    ax.set_xlabel('Total counts')
    ax.set_ylabel('Genes detected')
    ax.set_title('Counts vs Genes')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats = qc_result.overall_stats
    text = f"Total cells: {stats['n_cells']}\n"
    text += f"Clusters: {stats['n_clusters']}\n"
    text += f"Mean cluster size: {stats['mean_cluster_size']:.1f}\n"
    text += f"Min cluster size: {stats['min_cluster_size']}\n"
    text += f"Max cluster size: {stats['max_cluster_size']}\n"
    text += f"Mean counts/cell: {stats['mean_total_counts']:.1f}\n"
    text += f"Mean genes/cell: {stats['mean_n_genes']:.1f}"
    ax.text(0.1, 0.5, text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.set_title('Summary Statistics')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_batch_distribution(
    labels: np.ndarray,
    batch: np.ndarray,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot batch distribution across clusters.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    batch : np.ndarray
        Batch labels
    normalize : bool
        Normalize to proportions within each cluster
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    unique_clusters = sorted(np.unique(labels))
    unique_batches = sorted(np.unique(batch))

    # Create contingency table
    data = np.zeros((len(unique_clusters), len(unique_batches)))

    for i, c in enumerate(unique_clusters):
        cluster_mask = labels == c
        for j, b in enumerate(unique_batches):
            data[i, j] = np.sum(batch[cluster_mask] == b)

    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = data / np.maximum(row_sums, 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Stacked bar chart
    ax = axes[0]
    x = np.arange(len(unique_clusters))
    bottom = np.zeros(len(unique_clusters))

    for j, b in enumerate(unique_batches):
        ax.bar(x, data[:, j], bottom=bottom, label=str(b))
        bottom += data[:, j]

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_title('Batch Distribution per Cluster')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in unique_clusters])
    ax.legend(title='Batch', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Heatmap
    ax = axes[1]
    im = ax.imshow(data.T, aspect='auto', cmap='Blues')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Batch')
    ax.set_xticks(range(len(unique_clusters)))
    ax.set_xticklabels([str(c) for c in unique_clusters])
    ax.set_yticks(range(len(unique_batches)))
    ax.set_yticklabels([str(b) for b in unique_batches])
    ax.set_title('Batch-Cluster Heatmap')
    plt.colorbar(im, ax=ax, label='Proportion' if normalize else 'Count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_batch_umap(
    embedding: np.ndarray,
    batch: np.ndarray,
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot UMAP colored by batch and clusters.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates
    batch : np.ndarray
        Batch labels
    labels : np.ndarray, optional
        Cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    n_plots = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))

    if n_plots == 1:
        axes = [axes]

    # Color by batch
    ax = axes[0]
    unique_batches = np.unique(batch)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_batches)))

    for i, b in enumerate(unique_batches):
        mask = batch == b
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[i]], label=str(b), alpha=0.5, s=10
        )

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Colored by Batch')
    ax.legend(title='Batch', markerscale=2)

    # Color by cluster
    if labels is not None:
        ax = axes[1]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab20', alpha=0.5, s=10
        )

        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Colored by Cluster')
        plt.colorbar(scatter, ax=ax, label='Cluster')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def batch_effect_test(
    embedding: np.ndarray,
    batch: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000
) -> Dict:
    """
    Test for batch effects using permutation test.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates
    batch : np.ndarray
        Batch labels
    labels : np.ndarray
        Cluster labels
    n_permutations : int
        Number of permutations

    Returns
    -------
    Dict
        Test results including p-value
    """
    unique_clusters = np.unique(labels)
    unique_batches = np.unique(batch)

    # Create contingency table
    observed = np.zeros((len(unique_clusters), len(unique_batches)))

    for i, c in enumerate(unique_clusters):
        cluster_mask = labels == c
        for j, b in enumerate(unique_batches):
            observed[i, j] = np.sum(batch[cluster_mask] == b)

    # Chi-squared statistic
    _, observed_stat, _, _ = stats.chi2_contingency(observed)

    # Permutation test
    permuted_stats = []

    for _ in range(n_permutations):
        perm_batch = np.random.permutation(batch)

        perm_table = np.zeros((len(unique_clusters), len(unique_batches)))
        for i, c in enumerate(unique_clusters):
            cluster_mask = labels == c
            for j, b in enumerate(unique_batches):
                perm_table[i, j] = np.sum(perm_batch[cluster_mask] == b)

        _, perm_stat, _, _ = stats.chi2_contingency(perm_table)
        permuted_stats.append(perm_stat)

    permuted_stats = np.array(permuted_stats)
    pvalue = np.mean(permuted_stats >= observed_stat)

    return {
        'statistic': observed_stat,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'interpretation': 'Significant batch effect' if pvalue < 0.05 else 'No significant batch effect'
    }
