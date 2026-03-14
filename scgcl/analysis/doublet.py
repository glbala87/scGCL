"""Doublet detection for single-cell data."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
import warnings


@dataclass
class DoubletResult:
    """Container for doublet detection results."""

    doublet_scores: np.ndarray
    predicted_doublets: np.ndarray
    threshold: float
    n_doublets: int
    doublet_rate: float
    n_simulated: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'doublet_score': self.doublet_scores,
            'is_doublet': self.predicted_doublets
        })

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'n_cells': len(self.doublet_scores),
            'n_doublets': self.n_doublets,
            'doublet_rate': self.doublet_rate,
            'threshold': self.threshold,
            'n_simulated': self.n_simulated,
            'mean_score': float(np.mean(self.doublet_scores)),
            'median_score': float(np.median(self.doublet_scores))
        }


def detect_doublets(
    expression: np.ndarray,
    expected_doublet_rate: float = 0.05,
    n_neighbors: int = 30,
    n_simulated: Optional[int] = None,
    random_state: int = 42
) -> DoubletResult:
    """
    Detect doublets using a simulation-based approach.

    Creates synthetic doublets by averaging pairs of cells and scores
    real cells by their similarity to simulated doublets.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes), ideally PCA-reduced
    expected_doublet_rate : float
        Expected doublet rate (default: 5%)
    n_neighbors : int
        Number of neighbors for scoring
    n_simulated : int, optional
        Number of simulated doublets (default: 2x n_cells)
    random_state : int
        Random seed

    Returns
    -------
    DoubletResult
        Doublet detection results
    """
    np.random.seed(random_state)

    n_cells = expression.shape[0]

    if n_simulated is None:
        n_simulated = min(n_cells * 2, 10000)

    # Generate simulated doublets
    parent1_idx = np.random.randint(0, n_cells, n_simulated)
    parent2_idx = np.random.randint(0, n_cells, n_simulated)

    # Ensure different parents
    same_mask = parent1_idx == parent2_idx
    while np.any(same_mask):
        parent2_idx[same_mask] = np.random.randint(0, n_cells, np.sum(same_mask))
        same_mask = parent1_idx == parent2_idx

    # Create synthetic doublets (average of two cells)
    simulated_doublets = (expression[parent1_idx] + expression[parent2_idx]) / 2

    # Combine real cells and simulated doublets
    combined = np.vstack([expression, simulated_doublets])

    # Compute distances
    distances = cdist(combined, combined, metric='euclidean')

    # For each cell, count neighbors that are simulated doublets
    doublet_scores = np.zeros(n_cells)

    for i in range(n_cells):
        # Get k nearest neighbors (excluding self)
        dists = distances[i]
        dists[i] = np.inf  # Exclude self
        neighbor_idx = np.argsort(dists)[:n_neighbors]

        # Count how many neighbors are simulated doublets
        n_doublet_neighbors = np.sum(neighbor_idx >= n_cells)
        doublet_scores[i] = n_doublet_neighbors / n_neighbors

    # Determine threshold
    # Use expected doublet rate to set threshold
    threshold = np.percentile(doublet_scores, 100 * (1 - expected_doublet_rate))

    # Ensure threshold is reasonable
    threshold = max(threshold, 0.1)

    predicted_doublets = doublet_scores > threshold

    return DoubletResult(
        doublet_scores=doublet_scores,
        predicted_doublets=predicted_doublets,
        threshold=threshold,
        n_doublets=int(np.sum(predicted_doublets)),
        doublet_rate=float(np.mean(predicted_doublets)),
        n_simulated=n_simulated
    )


def detect_doublets_scrublet(
    counts: np.ndarray,
    expected_doublet_rate: float = 0.05,
    min_counts: int = 2,
    min_cells: int = 3,
    min_gene_variability_pctl: float = 85,
    n_prin_comps: int = 30,
    random_state: int = 42
) -> DoubletResult:
    """
    Detect doublets using Scrublet-like algorithm.

    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (cells x genes)
    expected_doublet_rate : float
        Expected doublet rate
    min_counts : int
        Minimum counts per gene
    min_cells : int
        Minimum cells per gene
    min_gene_variability_pctl : float
        Percentile for variable gene selection
    n_prin_comps : int
        Number of principal components
    random_state : int
        Random seed

    Returns
    -------
    DoubletResult
        Doublet detection results
    """
    np.random.seed(random_state)

    n_cells, n_genes = counts.shape

    # Filter genes
    gene_counts = np.sum(counts > 0, axis=0)
    gene_totals = np.sum(counts, axis=0)
    gene_mask = (gene_counts >= min_cells) & (gene_totals >= min_counts)

    if np.sum(gene_mask) < 100:
        warnings.warn("Too few genes after filtering, using all genes")
        gene_mask = np.ones(n_genes, dtype=bool)

    filtered_counts = counts[:, gene_mask]

    # Normalize
    cell_totals = np.sum(filtered_counts, axis=1, keepdims=True)
    cell_totals = np.maximum(cell_totals, 1)  # Avoid division by zero
    normalized = filtered_counts / cell_totals * 10000

    # Log transform
    log_normalized = np.log1p(normalized)

    # Highly variable genes
    gene_means = np.mean(log_normalized, axis=0)
    gene_vars = np.var(log_normalized, axis=0)

    # Select highly variable
    mean_bins = pd.cut(gene_means, bins=20, labels=False)
    gene_dispersion = np.zeros(len(gene_means))

    for b in range(20):
        mask = mean_bins == b
        if np.sum(mask) > 0:
            expected_var = np.mean(gene_vars[mask])
            if expected_var > 0:
                gene_dispersion[mask] = gene_vars[mask] / expected_var

    hvg_threshold = np.percentile(gene_dispersion, min_gene_variability_pctl)
    hvg_mask = gene_dispersion > hvg_threshold

    if np.sum(hvg_mask) < 50:
        # Use top 1000 variable genes
        hvg_mask = gene_dispersion >= np.percentile(gene_dispersion, 50)

    hvg_data = log_normalized[:, hvg_mask]

    # PCA
    hvg_centered = hvg_data - np.mean(hvg_data, axis=0)
    n_comps = min(n_prin_comps, hvg_centered.shape[0] - 1, hvg_centered.shape[1] - 1)

    try:
        from scipy.linalg import svd
        U, S, Vt = svd(hvg_centered, full_matrices=False)
        pca_embedding = U[:, :n_comps] * S[:n_comps]
    except Exception:
        # Fallback to simple approach
        pca_embedding = hvg_centered[:, :n_comps]

    # Run detection on PCA embedding
    return detect_doublets(
        pca_embedding,
        expected_doublet_rate=expected_doublet_rate,
        random_state=random_state
    )


def detect_doublets_adata(
    adata,
    use_rep: str = 'X_pca',
    expected_doublet_rate: float = 0.05,
    n_neighbors: int = 30,
    copy: bool = False
):
    """
    Detect doublets for AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    use_rep : str
        Representation to use (key in adata.obsm)
    expected_doublet_rate : float
        Expected doublet rate
    n_neighbors : int
        Number of neighbors
    copy : bool
        Return a copy

    Returns
    -------
    adata or None
        If copy=True, returns modified AnnData
    """
    if copy:
        adata = adata.copy()

    if use_rep in adata.obsm:
        X = adata.obsm[use_rep]
    else:
        warnings.warn(f"{use_rep} not found, using X")
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

    result = detect_doublets(
        X,
        expected_doublet_rate=expected_doublet_rate,
        n_neighbors=n_neighbors
    )

    # Add to adata
    adata.obs['doublet_score'] = result.doublet_scores
    adata.obs['predicted_doublet'] = result.predicted_doublets

    adata.uns['doublet_detection'] = {
        'threshold': result.threshold,
        'n_doublets': result.n_doublets,
        'doublet_rate': result.doublet_rate,
        'n_simulated': result.n_simulated
    }

    if copy:
        return adata


def plot_doublet_scores(
    result: Union[DoubletResult, 'AnnData'],
    embedding: Optional[np.ndarray] = None,
    embedding_key: str = 'X_umap',
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot doublet detection results.

    Parameters
    ----------
    result : DoubletResult or AnnData
        Doublet detection results
    embedding : np.ndarray, optional
        2D embedding coordinates
    embedding_key : str
        Key for embedding in adata.obsm
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    # Extract data
    if hasattr(result, 'doublet_scores'):
        scores = result.doublet_scores
        predicted = result.predicted_doublets
        threshold = result.threshold
    else:
        # AnnData
        scores = result.obs['doublet_score'].values
        predicted = result.obs['predicted_doublet'].values
        threshold = result.uns.get('doublet_detection', {}).get('threshold', 0.5)
        if embedding is None and embedding_key in result.obsm:
            embedding = result.obsm[embedding_key]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Score distribution
    ax = axes[0]
    ax.hist(scores, bins=50, density=True, alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    ax.set_xlabel('Doublet Score')
    ax.set_ylabel('Density')
    ax.set_title('Doublet Score Distribution')
    ax.legend()

    # Doublet counts
    ax = axes[1]
    n_doublets = np.sum(predicted)
    n_singlets = len(predicted) - n_doublets
    ax.bar(['Singlets', 'Doublets'], [n_singlets, n_doublets],
           color=['#1f77b4', '#d62728'])
    ax.set_ylabel('Number of cells')
    ax.set_title(f'Predicted Doublets: {n_doublets} ({100 * n_doublets / len(predicted):.1f}%)')

    # Embedding colored by doublet score
    ax = axes[2]
    if embedding is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=scores, cmap='Reds', alpha=0.5, s=10
        )
        # Mark predicted doublets
        doublet_mask = predicted.astype(bool)
        ax.scatter(
            embedding[doublet_mask, 0], embedding[doublet_mask, 1],
            c='red', marker='x', s=20, alpha=0.8, label='Doublet'
        )
        plt.colorbar(scatter, ax=ax, label='Doublet Score')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Doublet Scores')
    else:
        ax.text(0.5, 0.5, 'No embedding provided',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Embedding not available')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def filter_doublets(
    expression: np.ndarray,
    doublet_result: DoubletResult,
    return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Filter out predicted doublets from expression matrix.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    doublet_result : DoubletResult
        Results from detect_doublets
    return_mask : bool
        Also return the singlet mask

    Returns
    -------
    np.ndarray or tuple
        Filtered expression matrix, optionally with mask
    """
    singlet_mask = ~doublet_result.predicted_doublets
    filtered = expression[singlet_mask]

    if return_mask:
        return filtered, singlet_mask
    return filtered
