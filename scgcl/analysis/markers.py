"""Marker gene detection for cluster characterization."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarkerGeneResult:
    """Container for marker gene analysis results."""

    # Per-cluster results
    names: Dict[int, List[str]] = field(default_factory=dict)
    scores: Dict[int, np.ndarray] = field(default_factory=dict)
    pvals: Dict[int, np.ndarray] = field(default_factory=dict)
    pvals_adj: Dict[int, np.ndarray] = field(default_factory=dict)
    logfoldchanges: Dict[int, np.ndarray] = field(default_factory=dict)

    # Metadata
    n_genes: int = 0
    method: str = "wilcoxon"

    def to_dataframe(self, cluster: Optional[int] = None) -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        cluster : int, optional
            Specific cluster to return. If None, returns all clusters.

        Returns
        -------
        pd.DataFrame
            DataFrame with marker gene statistics
        """
        if cluster is not None:
            clusters = [cluster]
        else:
            clusters = sorted(self.names.keys())

        dfs = []
        for c in clusters:
            df = pd.DataFrame({
                'gene': self.names[c],
                'score': self.scores[c],
                'pval': self.pvals[c],
                'pval_adj': self.pvals_adj[c],
                'logfoldchange': self.logfoldchanges[c],
                'cluster': c
            })
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def top_markers(self, n: int = 10, cluster: Optional[int] = None) -> pd.DataFrame:
        """
        Get top N markers per cluster.

        Parameters
        ----------
        n : int
            Number of top markers to return
        cluster : int, optional
            Specific cluster. If None, returns for all clusters.

        Returns
        -------
        pd.DataFrame
            Top markers sorted by score
        """
        df = self.to_dataframe(cluster)

        result_dfs = []
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster].nlargest(n, 'score')
            result_dfs.append(cluster_df)

        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True)
        return pd.DataFrame(columns=df.columns)


def _wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """Perform Wilcoxon rank-sum test."""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0, 1.0

    try:
        stat, pval = stats.mannwhitneyu(
            group1, group2, alternative='two-sided'
        )
        # Convert to z-score
        n1, n2 = len(group1), len(group2)
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (stat - mu) / sigma if sigma > 0 else 0
        return z_score, pval
    except Exception:
        return 0.0, 1.0


def _ttest(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """Perform t-test."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0, 1.0

    try:
        stat, pval = stats.ttest_ind(group1, group2)
        return stat if not np.isnan(stat) else 0.0, pval if not np.isnan(pval) else 1.0
    except Exception:
        return 0.0, 1.0


def _logfoldchange(
    group1: np.ndarray,
    group2: np.ndarray,
    pseudocount: float = 1.0
) -> float:
    """Compute log2 fold change."""
    mean1 = np.mean(group1) + pseudocount
    mean2 = np.mean(group2) + pseudocount
    return np.log2(mean1 / mean2)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH correction
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        adjusted[i] = min(cummin, sorted_pvals[i] * n / (i + 1))
        cummin = adjusted[i]

    # Restore original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return np.clip(result, 0, 1)


def rank_genes_groups(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: Optional[List[str]] = None,
    method: str = 'wilcoxon',
    n_genes: Optional[int] = None,
    min_fold_change: float = 0.0,
    min_in_group_fraction: float = 0.0,
    reference: str = 'rest'
) -> MarkerGeneResult:
    """
    Rank genes for characterizing clusters.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray
        Cluster labels for each cell
    gene_names : list of str, optional
        Gene names. If None, uses indices.
    method : str
        Statistical method: 'wilcoxon', 't-test'
    n_genes : int, optional
        Number of genes to keep per cluster. If None, keeps all.
    min_fold_change : float
        Minimum log2 fold change threshold
    min_in_group_fraction : float
        Minimum fraction of cells in cluster expressing the gene
    reference : str
        Reference group: 'rest' (all other cells) or specific cluster label

    Returns
    -------
    MarkerGeneResult
        Container with marker gene statistics
    """
    n_cells, n_features = X.shape
    unique_labels = np.unique(labels)

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_features)]

    gene_names = np.array(gene_names)

    # Select test function
    test_fn = _wilcoxon_test if method == 'wilcoxon' else _ttest

    result = MarkerGeneResult(n_genes=n_features, method=method)

    for cluster in unique_labels:
        cluster_mask = labels == cluster
        other_mask = ~cluster_mask

        n_cluster = np.sum(cluster_mask)
        n_other = np.sum(other_mask)

        if n_cluster == 0 or n_other == 0:
            continue

        scores = np.zeros(n_features)
        pvals = np.ones(n_features)
        logfc = np.zeros(n_features)

        for gene_idx in range(n_features):
            expr_cluster = X[cluster_mask, gene_idx]
            expr_other = X[other_mask, gene_idx]

            # Compute statistics
            score, pval = test_fn(expr_cluster, expr_other)
            lfc = _logfoldchange(expr_cluster, expr_other)

            # Check minimum expression fraction
            in_group_frac = np.mean(expr_cluster > 0)

            if in_group_frac >= min_in_group_fraction and abs(lfc) >= min_fold_change:
                scores[gene_idx] = score
                pvals[gene_idx] = pval
                logfc[gene_idx] = lfc

        # Adjust p-values
        pvals_adj = _benjamini_hochberg(pvals)

        # Sort by score (descending)
        sorted_idx = np.argsort(-scores)

        if n_genes is not None:
            sorted_idx = sorted_idx[:n_genes]

        result.names[cluster] = gene_names[sorted_idx].tolist()
        result.scores[cluster] = scores[sorted_idx]
        result.pvals[cluster] = pvals[sorted_idx]
        result.pvals_adj[cluster] = pvals_adj[sorted_idx]
        result.logfoldchanges[cluster] = logfc[sorted_idx]

    return result


def find_marker_genes(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_markers: int = 10,
    method: str = 'wilcoxon',
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25
) -> pd.DataFrame:
    """
    Find marker genes for each cluster.

    This is a convenience wrapper around rank_genes_groups that returns
    a filtered DataFrame of significant markers.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray
        Cluster labels
    gene_names : list of str, optional
        Gene names
    n_markers : int
        Maximum number of markers per cluster
    method : str
        Statistical test: 'wilcoxon' or 't-test'
    pval_cutoff : float
        Adjusted p-value cutoff
    logfc_cutoff : float
        Minimum absolute log2 fold change

    Returns
    -------
    pd.DataFrame
        Marker genes with statistics
    """
    result = rank_genes_groups(
        X, labels, gene_names,
        method=method,
        min_fold_change=logfc_cutoff
    )

    df = result.to_dataframe()

    # Filter by significance
    df = df[
        (df['pval_adj'] < pval_cutoff) &
        (df['logfoldchange'].abs() >= logfc_cutoff)
    ]

    # Get top markers per cluster
    result_dfs = []
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster].nlargest(n_markers, 'score')
        result_dfs.append(cluster_df)

    if result_dfs:
        df = pd.concat(result_dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)

    return df.sort_values(['cluster', 'score'], ascending=[True, False])


def filter_markers(
    markers: Union[MarkerGeneResult, pd.DataFrame],
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
    n_top: Optional[int] = None
) -> pd.DataFrame:
    """
    Filter marker genes by significance thresholds.

    Parameters
    ----------
    markers : MarkerGeneResult or pd.DataFrame
        Marker gene results
    pval_cutoff : float
        Maximum adjusted p-value
    logfc_cutoff : float
        Minimum absolute log2 fold change
    n_top : int, optional
        Keep only top N per cluster

    Returns
    -------
    pd.DataFrame
        Filtered marker genes
    """
    if isinstance(markers, MarkerGeneResult):
        df = markers.to_dataframe()
    else:
        df = markers.copy()

    # Apply filters
    mask = (
        (df['pval_adj'] < pval_cutoff) &
        (df['logfoldchange'].abs() >= logfc_cutoff)
    )
    df = df[mask]

    if n_top is not None:
        result_dfs = []
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster].nlargest(n_top, 'score')
            result_dfs.append(cluster_df)
        if result_dfs:
            df = pd.concat(result_dfs, ignore_index=True)
        else:
            df = pd.DataFrame(columns=df.columns)

    return df


def plot_markers(
    markers: Union[MarkerGeneResult, pd.DataFrame],
    n_genes: int = 5,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot top marker genes as a heatmap-style visualization.

    Parameters
    ----------
    markers : MarkerGeneResult or pd.DataFrame
        Marker gene results
    n_genes : int
        Number of genes per cluster to show
    figsize : tuple
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    if isinstance(markers, MarkerGeneResult):
        df = markers.top_markers(n=n_genes)
    else:
        df = markers.groupby('cluster').head(n_genes)

    clusters = sorted(df['cluster'].unique())

    fig, axes = plt.subplots(1, len(clusters), figsize=figsize, sharey=False)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        cluster_df = df[df['cluster'] == cluster].head(n_genes)

        y_pos = np.arange(len(cluster_df))
        ax.barh(y_pos, cluster_df['score'], color=f'C{cluster}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cluster_df['gene'])
        ax.set_xlabel('Score')
        ax.set_title(f'Cluster {cluster}')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
