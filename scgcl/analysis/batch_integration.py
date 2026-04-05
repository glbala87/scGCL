"""Batch integration and correction methods for single-cell data."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.linalg import svd, lstsq
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Container for batch integration results."""

    corrected: np.ndarray
    batch_labels: np.ndarray
    method: str
    metrics: Dict

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "Batch Integration Results",
            "=" * 40,
            f"Method: {self.method}",
            f"Number of batches: {len(np.unique(self.batch_labels))}",
            f"Output shape: {self.corrected.shape}",
            "",
            "Metrics:",
        ]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def harmony(
    embedding: np.ndarray,
    batch: np.ndarray,
    n_clusters: int = 100,
    max_iter: int = 20,
    theta: float = 2.0,
    sigma: float = 0.1,
    block_size: float = 0.05,
    verbose: bool = False
) -> IntegrationResult:
    """
    Harmony batch integration algorithm.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings (cells x dims), typically PCA
    batch : np.ndarray
        Batch labels for each cell
    n_clusters : int
        Number of clusters for soft clustering
    max_iter : int
        Maximum iterations
    theta : float
        Diversity clustering penalty
    sigma : float
        Width of soft kmeans clusters
    block_size : float
        Proportion of cells to update per iteration
    verbose : bool
        Print progress

    Returns
    -------
    IntegrationResult
        Integration results with corrected embedding
    """
    n_cells, n_dims = embedding.shape
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Encode batches as integers
    batch_int = np.zeros(n_cells, dtype=int)
    for i, b in enumerate(unique_batches):
        batch_int[batch == b] = i

    # Initialize: normalize embeddings
    Z = embedding.copy()
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-10)

    # Initialize cluster centroids with kmeans++
    n_clusters = min(n_clusters, n_cells // 2)
    centroids = _kmeans_plus_plus_init(Z, n_clusters)

    # Compute batch proportions
    batch_props = np.array([np.mean(batch_int == b) for b in range(n_batches)])

    for iteration in range(max_iter):
        if verbose:
            logger.info("Harmony iteration %d/%d", iteration + 1, max_iter)

        # E-step: soft cluster assignment
        distances = cdist(Z, centroids, metric='euclidean')
        R = np.exp(-distances ** 2 / (2 * sigma ** 2))
        R = R / (R.sum(axis=1, keepdims=True) + 1e-10)

        # Compute cluster-batch frequencies
        cluster_batch_counts = np.zeros((n_clusters, n_batches))
        for k in range(n_clusters):
            for b in range(n_batches):
                cluster_batch_counts[k, b] = R[batch_int == b, k].sum()

        # Diversity penalty: penalize clusters dominated by one batch
        cluster_batch_props = cluster_batch_counts / (cluster_batch_counts.sum(axis=1, keepdims=True) + 1e-10)
        diversity_penalty = 1 - theta * np.abs(cluster_batch_props - batch_props)
        diversity_penalty = np.clip(diversity_penalty, 0.1, 1)

        # Apply diversity penalty
        for b in range(n_batches):
            R[batch_int == b] *= diversity_penalty[:, b]
        R = R / (R.sum(axis=1, keepdims=True) + 1e-10)

        # M-step: update centroids
        for k in range(n_clusters):
            weights = R[:, k]
            if weights.sum() > 0:
                centroids[k] = (weights[:, np.newaxis] * Z).sum(axis=0) / weights.sum()

        # Correction step: adjust embeddings
        Z_new = embedding.copy()

        for b in range(n_batches):
            batch_mask = batch_int == b

            # Compute batch-specific centroid offsets
            for k in range(n_clusters):
                batch_cluster_mean = np.average(
                    embedding[batch_mask], axis=0,
                    weights=R[batch_mask, k] + 1e-10
                )
                global_cluster_mean = np.average(
                    embedding, axis=0,
                    weights=R[:, k] + 1e-10
                )

                # Correction for this batch-cluster
                correction = global_cluster_mean - batch_cluster_mean

                # Apply weighted correction
                Z_new[batch_mask] += R[batch_mask, k:k+1] * correction

        Z = Z_new
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-10)

    # Compute integration metrics
    metrics = _compute_integration_metrics(Z, batch_int)

    return IntegrationResult(
        corrected=Z,
        batch_labels=batch,
        method='harmony',
        metrics=metrics
    )


def _kmeans_plus_plus_init(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """K-means++ initialization."""
    n_samples = X.shape[0]
    centroids = [X[np.random.randint(n_samples)]]

    for _ in range(n_clusters - 1):
        distances = cdist(X, np.array(centroids), metric='euclidean')
        min_distances = distances.min(axis=1)
        probs = min_distances ** 2
        probs /= probs.sum()
        new_centroid_idx = np.random.choice(n_samples, p=probs)
        centroids.append(X[new_centroid_idx])

    return np.array(centroids)


def mnn_correct(
    data_list: List[np.ndarray],
    batch_labels: List,
    k: int = 20,
    sigma: float = 1.0
) -> IntegrationResult:
    """
    Mutual Nearest Neighbors (MNN) batch correction.

    Parameters
    ----------
    data_list : List[np.ndarray]
        List of expression matrices per batch
    batch_labels : List
        Labels for each batch
    k : int
        Number of nearest neighbors
    sigma : float
        Gaussian kernel bandwidth

    Returns
    -------
    IntegrationResult
        Integration results
    """
    n_batches = len(data_list)

    if n_batches < 2:
        raise ValueError("Need at least 2 batches for MNN correction")

    # Start with first batch as reference
    corrected = data_list[0].copy()
    all_batches = np.array([batch_labels[0]] * len(data_list[0]))

    for i in range(1, n_batches):
        query = data_list[i]

        # Find MNN pairs
        mnn_pairs = _find_mnn(corrected, query, k)

        if len(mnn_pairs) == 0:
            warnings.warn(f"No MNN pairs found for batch {batch_labels[i]}")
            corrected = np.vstack([corrected, query])
            all_batches = np.concatenate([all_batches, [batch_labels[i]] * len(query)])
            continue

        # Compute correction vectors
        ref_idx, query_idx = zip(*mnn_pairs)
        ref_idx = np.array(ref_idx)
        query_idx = np.array(query_idx)

        # Average correction vector
        corrections = corrected[ref_idx] - query[query_idx]
        mean_correction = corrections.mean(axis=0)

        # Apply Gaussian-weighted correction
        query_corrected = query.copy()
        for j in range(len(query)):
            # Distance to MNN pairs in query
            dists = np.linalg.norm(query[query_idx] - query[j], axis=1)
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights /= weights.sum() + 1e-10

            # Weighted average of correction vectors
            weighted_correction = (weights[:, np.newaxis] * corrections).sum(axis=0)
            query_corrected[j] += weighted_correction

        corrected = np.vstack([corrected, query_corrected])
        all_batches = np.concatenate([all_batches, [batch_labels[i]] * len(query)])

    # Compute metrics
    batch_int = np.zeros(len(all_batches), dtype=int)
    unique_batches = np.unique(all_batches)
    for i, b in enumerate(unique_batches):
        batch_int[all_batches == b] = i

    metrics = _compute_integration_metrics(corrected, batch_int)

    return IntegrationResult(
        corrected=corrected,
        batch_labels=all_batches,
        method='mnn',
        metrics=metrics
    )


def _find_mnn(ref: np.ndarray, query: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Find mutual nearest neighbor pairs."""
    # Query -> Ref neighbors
    dist_q2r = cdist(query, ref, metric='euclidean')
    nn_q2r = np.argsort(dist_q2r, axis=1)[:, :k]

    # Ref -> Query neighbors
    dist_r2q = dist_q2r.T
    nn_r2q = np.argsort(dist_r2q, axis=1)[:, :k]

    # Find mutual pairs
    mnn_pairs = []
    for i in range(len(query)):
        for j in nn_q2r[i]:
            if i in nn_r2q[j]:
                mnn_pairs.append((j, i))

    return mnn_pairs


def combat(
    expression: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[np.ndarray] = None
) -> IntegrationResult:
    """
    ComBat batch effect correction.

    Empirical Bayes method for batch correction.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    batch : np.ndarray
        Batch labels
    covariates : np.ndarray, optional
        Additional covariates to preserve

    Returns
    -------
    IntegrationResult
        Integration results
    """
    n_cells, n_genes = expression.shape
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Encode batches
    batch_int = np.zeros(n_cells, dtype=int)
    for i, b in enumerate(unique_batches):
        batch_int[batch == b] = i

    # Design matrix
    batch_design = np.zeros((n_cells, n_batches))
    for i in range(n_batches):
        batch_design[batch_int == i, i] = 1

    # Add covariates if provided
    if covariates is not None:
        design = np.hstack([batch_design, covariates])
    else:
        design = batch_design

    # Standardize data
    grand_mean = expression.mean(axis=0)
    var_pooled = expression.var(axis=0)
    var_pooled = np.maximum(var_pooled, 1e-10)

    stand_expr = (expression - grand_mean) / np.sqrt(var_pooled)

    # Estimate batch effects
    batch_means = np.zeros((n_batches, n_genes))
    batch_vars = np.zeros((n_batches, n_genes))

    for i in range(n_batches):
        batch_mask = batch_int == i
        batch_means[i] = stand_expr[batch_mask].mean(axis=0)
        batch_vars[i] = stand_expr[batch_mask].var(axis=0)

    # Empirical Bayes estimation of batch parameters
    # Prior for means: normal
    gamma_bar = batch_means.mean(axis=0)
    tau2 = batch_means.var(axis=0)

    # Prior for variances: inverse gamma
    delta_bar = batch_vars.mean(axis=0)

    # Posterior estimates (simplified EB)
    gamma_star = batch_means  # Simplified: use sample estimates
    delta_star = batch_vars

    # Apply correction
    corrected = np.zeros_like(expression)

    for i in range(n_batches):
        batch_mask = batch_int == i
        batch_data = stand_expr[batch_mask]

        # Remove batch effect
        corrected_batch = (batch_data - gamma_star[i]) / np.sqrt(delta_star[i] + 1e-10)

        # Add back global mean and variance
        corrected_batch = corrected_batch * np.sqrt(var_pooled) + grand_mean

        corrected[batch_mask] = corrected_batch

    # Compute metrics
    metrics = _compute_integration_metrics(corrected, batch_int)

    return IntegrationResult(
        corrected=corrected,
        batch_labels=batch,
        method='combat',
        metrics=metrics
    )


def regress_batch(
    expression: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[np.ndarray] = None
) -> IntegrationResult:
    """
    Simple linear regression batch correction.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    batch : np.ndarray
        Batch labels
    covariates : np.ndarray, optional
        Covariates to preserve

    Returns
    -------
    IntegrationResult
        Integration results
    """
    n_cells, n_genes = expression.shape
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Encode batches
    batch_int = np.zeros(n_cells, dtype=int)
    for i, b in enumerate(unique_batches):
        batch_int[batch == b] = i

    # One-hot encode batches (drop first for identifiability)
    batch_design = np.zeros((n_cells, n_batches - 1))
    for i in range(1, n_batches):
        batch_design[batch_int == i, i - 1] = 1

    # Add intercept
    design = np.hstack([np.ones((n_cells, 1)), batch_design])

    # Add covariates if provided
    if covariates is not None:
        design = np.hstack([design, covariates])

    # Fit regression and remove batch effects
    corrected = np.zeros_like(expression)

    for j in range(n_genes):
        y = expression[:, j]
        beta = lstsq(design, y)[0]

        # Remove batch effect (keep intercept and covariates)
        batch_effect = design[:, 1:n_batches] @ beta[1:n_batches]
        corrected[:, j] = y - batch_effect

    # Compute metrics
    metrics = _compute_integration_metrics(corrected, batch_int)

    return IntegrationResult(
        corrected=corrected,
        batch_labels=batch,
        method='regression',
        metrics=metrics
    )


def integrate(
    expression: np.ndarray,
    batch: np.ndarray,
    method: str = 'harmony',
    **kwargs
) -> IntegrationResult:
    """
    Unified interface for batch integration.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix or embeddings
    batch : np.ndarray
        Batch labels
    method : str
        Method: 'harmony', 'combat', 'regression', 'mnn'
    **kwargs
        Method-specific parameters

    Returns
    -------
    IntegrationResult
        Integration results
    """
    if method == 'harmony':
        return harmony(expression, batch, **kwargs)
    elif method == 'combat':
        return combat(expression, batch, **kwargs)
    elif method == 'regression':
        return regress_batch(expression, batch, **kwargs)
    elif method == 'mnn':
        # Convert to list format
        unique_batches = np.unique(batch)
        data_list = [expression[batch == b] for b in unique_batches]
        return mnn_correct(data_list, unique_batches.tolist(), **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_integration_metrics(
    embedding: np.ndarray,
    batch: np.ndarray,
    n_neighbors: int = 30
) -> Dict:
    """Compute integration quality metrics."""
    n_cells = len(batch)
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    # Compute kNN
    distances = cdist(embedding, embedding, metric='euclidean')
    knn_indices = np.argsort(distances, axis=1)[:, 1:n_neighbors + 1]

    # Batch mixing entropy (LISI-like)
    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch[knn_indices[i]]
        _, counts = np.unique(neighbor_batches, return_counts=True)
        props = counts / n_neighbors
        entropy = -np.sum(props * np.log(props + 1e-10))
        max_entropy = np.log(n_batches)
        mixing_scores.append(entropy / max_entropy if max_entropy > 0 else 0)

    # Batch ASW (average silhouette width for batches - lower is better for mixing)
    from sklearn.metrics import silhouette_samples
    try:
        batch_asw = silhouette_samples(embedding, batch).mean()
    except Exception:
        batch_asw = 0

    # kBET-like acceptance rate
    # Fraction of cells whose neighborhood batch composition matches global
    global_props = np.array([np.mean(batch == b) for b in unique_batches])
    kbet_accept = 0
    for i in range(n_cells):
        neighbor_batches = batch[knn_indices[i]]
        local_props = np.array([np.mean(neighbor_batches == b) for b in unique_batches])
        # Chi-squared-like test
        chi2 = np.sum((local_props - global_props) ** 2 / (global_props + 1e-10))
        if chi2 < 0.05 * n_batches:  # Simplified acceptance
            kbet_accept += 1
    kbet_accept /= n_cells

    return {
        'mixing_score': np.mean(mixing_scores),
        'batch_asw': batch_asw,
        'kbet_acceptance': kbet_accept
    }


def compute_lisi(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 30
) -> np.ndarray:
    """
    Compute Local Inverse Simpson's Index (LISI).

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings
    labels : np.ndarray
        Labels to compute LISI for (e.g., batch or cell type)
    n_neighbors : int
        Number of neighbors

    Returns
    -------
    np.ndarray
        LISI score for each cell
    """
    n_cells = len(labels)
    unique_labels = np.unique(labels)

    distances = cdist(embedding, embedding, metric='euclidean')
    knn_indices = np.argsort(distances, axis=1)[:, 1:n_neighbors + 1]

    lisi = np.zeros(n_cells)

    for i in range(n_cells):
        neighbor_labels = labels[knn_indices[i]]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        props = counts / n_neighbors

        # Simpson's index
        simpson = np.sum(props ** 2)

        # Inverse Simpson's index
        lisi[i] = 1 / simpson if simpson > 0 else 1

    return lisi


def plot_integration(
    before: np.ndarray,
    after: np.ndarray,
    batch: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'pca',
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot before/after integration comparison.

    Parameters
    ----------
    before : np.ndarray
        Embeddings before integration
    after : np.ndarray
        Embeddings after integration
    batch : np.ndarray
        Batch labels
    labels : np.ndarray, optional
        Cell type labels
    method : str
        Dimensionality reduction: 'pca', 'umap'
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

    # Reduce dimensions if needed
    if before.shape[1] > 2:
        if method == 'pca':
            from scipy.linalg import svd
            before_2d = svd(before - before.mean(axis=0), full_matrices=False)[0][:, :2]
            after_2d = svd(after - after.mean(axis=0), full_matrices=False)[0][:, :2]
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                before_2d = reducer.fit_transform(before)
                after_2d = reducer.fit_transform(after)
            except ImportError:
                warnings.warn("umap not installed, using PCA")
                from scipy.linalg import svd
                before_2d = svd(before - before.mean(axis=0), full_matrices=False)[0][:, :2]
                after_2d = svd(after - after.mean(axis=0), full_matrices=False)[0][:, :2]
        else:
            before_2d = before[:, :2]
            after_2d = after[:, :2]
    else:
        before_2d = before
        after_2d = after

    n_cols = 2 if labels is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    unique_batches = np.unique(batch)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_batches)))

    # Before - by batch
    ax = axes[0]
    for i, b in enumerate(unique_batches):
        mask = batch == b
        ax.scatter(before_2d[mask, 0], before_2d[mask, 1],
                   c=[colors[i]], label=str(b), s=5, alpha=0.5)
    ax.set_title('Before Integration (Batch)')
    ax.legend(markerscale=3)

    # After - by batch
    ax = axes[1]
    for i, b in enumerate(unique_batches):
        mask = batch == b
        ax.scatter(after_2d[mask, 0], after_2d[mask, 1],
                   c=[colors[i]], label=str(b), s=5, alpha=0.5)
    ax.set_title('After Integration (Batch)')
    ax.legend(markerscale=3)

    # If labels provided, also show cell types
    if labels is not None:
        unique_labels = np.unique(labels)
        label_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        ax = axes[2]
        for i, l in enumerate(unique_labels):
            mask = labels == l
            ax.scatter(before_2d[mask, 0], before_2d[mask, 1],
                       c=[label_colors[i]], label=str(l), s=5, alpha=0.5)
        ax.set_title('Before Integration (Cell Type)')

        ax = axes[3]
        for i, l in enumerate(unique_labels):
            mask = labels == l
            ax.scatter(after_2d[mask, 0], after_2d[mask, 1],
                       c=[label_colors[i]], label=str(l), s=5, alpha=0.5)
        ax.set_title('After Integration (Cell Type)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
