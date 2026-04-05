"""Trajectory and pseudotime analysis for single-cell data."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import sparse
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.sparse import csr_matrix
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Container for trajectory analysis results."""

    pseudotime: np.ndarray
    branch_labels: Optional[np.ndarray]
    milestone_network: Optional[pd.DataFrame]
    root_cell: int
    n_branches: int
    method: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            'pseudotime': self.pseudotime
        })
        if self.branch_labels is not None:
            df['branch'] = self.branch_labels
        return df

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "Trajectory Analysis Results",
            "=" * 40,
            f"Method: {self.method}",
            f"Root cell: {self.root_cell}",
            f"Number of branches: {self.n_branches}",
            f"Pseudotime range: [{self.pseudotime.min():.3f}, {self.pseudotime.max():.3f}]",
        ]
        if self.branch_labels is not None:
            unique_branches = np.unique(self.branch_labels)
            lines.append(f"Branch sizes: {dict(zip(unique_branches, [np.sum(self.branch_labels == b) for b in unique_branches]))}")
        return "\n".join(lines)


def diffusion_pseudotime(
    embedding: np.ndarray,
    root: Optional[int] = None,
    n_neighbors: int = 15,
    n_components: int = 10,
    alpha: float = 1.0
) -> TrajectoryResult:
    """
    Compute diffusion pseudotime.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings (cells x dims)
    root : int, optional
        Root cell index (auto-detected if None)
    n_neighbors : int
        Number of neighbors for diffusion
    n_components : int
        Number of diffusion components
    alpha : float
        Diffusion time scale

    Returns
    -------
    TrajectoryResult
        Trajectory analysis results
    """
    n_cells = embedding.shape[0]

    # Build kNN graph
    distances = cdist(embedding, embedding, metric='euclidean')
    knn_graph = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        neighbors = np.argsort(distances[i])[:n_neighbors + 1]
        for j in neighbors:
            if i != j:
                knn_graph[i, j] = distances[i, j]
                knn_graph[j, i] = distances[j, i]

    # Convert to affinity matrix
    sigma = np.median(knn_graph[knn_graph > 0])
    affinity = np.exp(-knn_graph ** 2 / (2 * sigma ** 2))
    affinity[knn_graph == 0] = 0
    np.fill_diagonal(affinity, 0)

    # Normalize to transition matrix
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    transition = affinity / row_sums

    # Compute diffusion components via eigendecomposition
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(transition)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Use top components (excluding first trivial one)
        n_comps = min(n_components + 1, len(eigenvalues))
        diff_components = eigenvectors[:, 1:n_comps] * (eigenvalues[1:n_comps] ** alpha)
    except Exception:
        # Fallback: use PCA on transition matrix
        from scipy.linalg import svd
        U, S, Vt = svd(transition, full_matrices=False)
        n_comps = min(n_components, len(S))
        diff_components = U[:, :n_comps] * S[:n_comps]

    # Auto-detect root if not provided
    if root is None:
        # Use cell furthest from centroid in diffusion space
        centroid = diff_components.mean(axis=0)
        distances_to_centroid = np.linalg.norm(diff_components - centroid, axis=1)
        root = np.argmax(distances_to_centroid)

    # Compute pseudotime as distance from root in diffusion space
    root_coords = diff_components[root]
    pseudotime = np.linalg.norm(diff_components - root_coords, axis=1)

    # Normalize to [0, 1]
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)

    return TrajectoryResult(
        pseudotime=pseudotime,
        branch_labels=None,
        milestone_network=None,
        root_cell=root,
        n_branches=1,
        method='diffusion_pseudotime'
    )


def principal_curve(
    embedding: np.ndarray,
    n_points: int = 100,
    max_iter: int = 50,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a principal curve through the data.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings (cells x dims)
    n_points : int
        Number of points on the curve
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    pseudotime : np.ndarray
        Pseudotime for each cell
    curve : np.ndarray
        Points defining the principal curve
    """
    n_cells, n_dims = embedding.shape

    # Initialize with first principal component
    centered = embedding - embedding.mean(axis=0)
    try:
        from scipy.linalg import svd
        U, S, Vt = svd(centered, full_matrices=False)
        pc1 = centered @ Vt[0]
    except Exception:
        pc1 = centered[:, 0]

    # Project onto PC1 for initial ordering
    order = np.argsort(pc1)
    pseudotime = np.zeros(n_cells)
    pseudotime[order] = np.linspace(0, 1, n_cells)

    # Iteratively refine
    for iteration in range(max_iter):
        old_pseudotime = pseudotime.copy()

        # Fit smooth curve through ordered points
        sorted_idx = np.argsort(pseudotime)
        t_values = np.linspace(0, 1, n_points)

        # Simple smoothing: weighted average in windows
        curve = np.zeros((n_points, n_dims))
        for i, t in enumerate(t_values):
            weights = np.exp(-((pseudotime - t) ** 2) / 0.05)
            weights /= weights.sum() + 1e-10
            curve[i] = (embedding.T @ weights)

        # Project cells onto curve
        for i in range(n_cells):
            dists = np.linalg.norm(curve - embedding[i], axis=1)
            closest = np.argmin(dists)
            pseudotime[i] = t_values[closest]

        # Check convergence
        if np.max(np.abs(pseudotime - old_pseudotime)) < tol:
            break

    return pseudotime, curve


def slingshot(
    embedding: np.ndarray,
    labels: np.ndarray,
    start_cluster: Optional[int] = None,
    end_clusters: Optional[List[int]] = None
) -> TrajectoryResult:
    """
    Slingshot-like trajectory inference.

    Fits principal curves through cluster centroids.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings
    labels : np.ndarray
        Cluster labels
    start_cluster : int, optional
        Starting cluster (auto-detected if None)
    end_clusters : List[int], optional
        Terminal clusters

    Returns
    -------
    TrajectoryResult
        Trajectory results with branch assignments
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    # Compute cluster centroids
    centroids = np.array([
        embedding[labels == c].mean(axis=0)
        for c in unique_clusters
    ])

    # Build MST on centroids
    centroid_dists = squareform(pdist(centroids))
    mst = minimum_spanning_tree(csr_matrix(centroid_dists)).toarray()
    mst = mst + mst.T  # Make symmetric

    # Find start cluster if not provided
    if start_cluster is None:
        # Use cluster with highest average expression of first PC
        centered = embedding - embedding.mean(axis=0)
        try:
            from scipy.linalg import svd
            U, S, Vt = svd(centered, full_matrices=False)
            pc1_scores = centered @ Vt[0]
        except Exception:
            pc1_scores = centered[:, 0]

        cluster_pc1 = [pc1_scores[labels == c].mean() for c in unique_clusters]
        start_cluster = unique_clusters[np.argmin(cluster_pc1)]

    start_idx = np.where(unique_clusters == start_cluster)[0][0]

    # Find end clusters (leaves of MST)
    if end_clusters is None:
        degrees = (mst > 0).sum(axis=1)
        end_indices = np.where(degrees == 1)[0]
        end_indices = end_indices[end_indices != start_idx]
        end_clusters = unique_clusters[end_indices].tolist()

    # Compute paths from start to each end
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cost = mst.copy()
        cost[cost == 0] = np.inf
        dist_matrix, predecessors = dijkstra(
            csr_matrix(cost),
            directed=False,
            indices=start_idx,
            return_predecessors=True
        )

    # Assign cells to branches and compute pseudotime
    n_cells = len(labels)
    pseudotime = np.zeros(n_cells)
    branch_labels = np.zeros(n_cells, dtype=int)

    for branch_id, end_cluster in enumerate(end_clusters):
        end_idx = np.where(unique_clusters == end_cluster)[0][0]

        # Trace path from end to start
        path = [end_idx]
        current = end_idx
        while current != start_idx:
            current = predecessors[current]
            if current < 0:
                break
            path.append(current)
        path = path[::-1]  # Start to end

        # Assign cells in path clusters to this branch
        path_clusters = unique_clusters[path]
        for i, cluster in enumerate(path_clusters):
            mask = labels == cluster
            branch_labels[mask] = branch_id

            # Pseudotime based on position along path
            base_time = i / len(path_clusters)

            # Refine within cluster based on distance to next centroid
            if i < len(path_clusters) - 1:
                next_centroid = centroids[path[i + 1]]
                cluster_cells = embedding[mask]
                dists = np.linalg.norm(cluster_cells - next_centroid, axis=1)
                # Closer to next = higher pseudotime
                within_cluster_time = 1 - (dists - dists.min()) / (dists.max() - dists.min() + 1e-10)
                pseudotime[mask] = base_time + within_cluster_time / len(path_clusters)
            else:
                pseudotime[mask] = base_time + 0.5 / len(path_clusters)

    # Normalize pseudotime
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)

    # Create milestone network
    edges = []
    for branch_id, end_cluster in enumerate(end_clusters):
        end_idx = np.where(unique_clusters == end_cluster)[0][0]
        path = [end_idx]
        current = end_idx
        while current != start_idx:
            current = predecessors[current]
            if current < 0:
                break
            path.append(current)

        for i in range(len(path) - 1):
            edges.append({
                'from': unique_clusters[path[i + 1]],
                'to': unique_clusters[path[i]],
                'branch': branch_id
            })

    milestone_network = pd.DataFrame(edges) if edges else None

    return TrajectoryResult(
        pseudotime=pseudotime,
        branch_labels=branch_labels,
        milestone_network=milestone_network,
        root_cell=int(np.where(labels == start_cluster)[0][0]),
        n_branches=len(end_clusters),
        method='slingshot'
    )


def paga(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    threshold: float = 0.05,
    root: Optional[int] = None
) -> TrajectoryResult:
    """
    PAGA-like partition-based graph abstraction.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings
    labels : np.ndarray
        Cluster labels
    n_neighbors : int
        Number of neighbors for connectivity
    threshold : float
        Connectivity threshold
    root : int, optional
        Root cell index for pseudotime computation

    Returns
    -------
    TrajectoryResult
        Trajectory results with connectivity-based pseudotime
    """
    unique_clusters = sorted(np.unique(labels))
    n_clusters = len(unique_clusters)

    # Build kNN graph
    n_cells = embedding.shape[0]
    distances = cdist(embedding, embedding, metric='euclidean')

    knn_indices = np.zeros((n_cells, n_neighbors), dtype=int)
    for i in range(n_cells):
        knn_indices[i] = np.argsort(distances[i])[1:n_neighbors + 1]

    # Count inter-cluster edges
    connectivity = np.zeros((n_clusters, n_clusters))

    for i in range(n_cells):
        cluster_i = np.where(unique_clusters == labels[i])[0][0]
        for j in knn_indices[i]:
            cluster_j = np.where(unique_clusters == labels[j])[0][0]
            connectivity[cluster_i, cluster_j] += 1

    # Normalize by cluster sizes
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_clusters])

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                expected = cluster_sizes[i] * cluster_sizes[j] * n_neighbors / n_cells
                connectivity[i, j] = connectivity[i, j] / (expected + 1e-10)

    # Make symmetric
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)

    # Apply threshold
    connectivity[connectivity < threshold] = 0

    # Create edge list
    edges = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if connectivity[i, j] > 0:
                edges.append({
                    'from': unique_clusters[i],
                    'to': unique_clusters[j],
                    'weight': connectivity[i, j]
                })

    milestone_network = pd.DataFrame(edges) if edges else None

    # Compute pseudotime based on graph distances from root
    if root is None:
        # Use cell at minimum of first PC as root
        centered = embedding - embedding.mean(axis=0)
        pc1 = centered[:, 0]
        root = int(np.argmin(pc1))

    root_cluster_idx = np.where(unique_clusters == labels[root])[0][0]

    # Convert connectivity to distance (inverse weight)
    dist_matrix = np.zeros_like(connectivity)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if connectivity[i, j] > 0:
                dist_matrix[i, j] = 1.0 / (connectivity[i, j] + 1e-10)
            elif i != j:
                dist_matrix[i, j] = np.inf

    # Compute shortest paths from root cluster
    cluster_pseudotime, _ = dijkstra(
        csr_matrix(dist_matrix),
        directed=False,
        indices=root_cluster_idx,
        return_predecessors=True
    )

    # Handle unreachable clusters
    cluster_pseudotime[np.isinf(cluster_pseudotime)] = cluster_pseudotime[~np.isinf(cluster_pseudotime)].max()

    # Normalize cluster pseudotime
    if cluster_pseudotime.max() > 0:
        cluster_pseudotime = cluster_pseudotime / cluster_pseudotime.max()

    # Assign pseudotime to cells based on cluster
    pseudotime = np.zeros(n_cells)
    for i, c in enumerate(unique_clusters):
        mask = labels == c
        base_time = cluster_pseudotime[i]
        # Add small variation within cluster based on distance to cluster center
        cluster_embedding = embedding[mask]
        centroid = cluster_embedding.mean(axis=0)
        within_dists = np.linalg.norm(cluster_embedding - centroid, axis=1)
        if within_dists.max() > 0:
            within_variation = within_dists / within_dists.max() * 0.05
        else:
            within_variation = 0
        pseudotime[mask] = base_time + within_variation

    # Normalize final pseudotime
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)

    return TrajectoryResult(
        pseudotime=pseudotime,
        branch_labels=None,
        milestone_network=milestone_network,
        root_cell=root,
        n_branches=1,
        method='paga'
    )


def infer_trajectory(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'dpt',
    root: Optional[int] = None,
    **kwargs
) -> TrajectoryResult:
    """
    Unified interface for trajectory inference.

    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings
    labels : np.ndarray, optional
        Cluster labels (required for slingshot and paga)
    method : str
        Method: 'dpt' (diffusion pseudotime), 'slingshot', 'principal_curve', 'paga'
    root : int, optional
        Root cell index
    **kwargs
        Method-specific parameters

    Returns
    -------
    TrajectoryResult
        Trajectory results
    """
    if method == 'dpt' or method == 'diffusion_pseudotime':
        return diffusion_pseudotime(embedding, root=root, **kwargs)

    elif method == 'slingshot':
        if labels is None:
            raise ValueError("labels required for slingshot method")
        return slingshot(embedding, labels, **kwargs)

    elif method == 'paga':
        if labels is None:
            raise ValueError("labels required for paga method")
        return paga(embedding, labels, root=root, **kwargs)

    elif method == 'principal_curve':
        pseudotime, curve = principal_curve(embedding, **kwargs)
        if root is not None:
            # Flip if root has high pseudotime
            if pseudotime[root] > 0.5:
                pseudotime = 1 - pseudotime
        return TrajectoryResult(
            pseudotime=pseudotime,
            branch_labels=None,
            milestone_network=None,
            root_cell=root if root is not None else int(np.argmin(pseudotime)),
            n_branches=1,
            method='principal_curve'
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_trajectory(
    embedding: np.ndarray,
    result: TrajectoryResult,
    labels: Optional[np.ndarray] = None,
    show_arrows: bool = True,
    arrow_density: float = 0.1,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot trajectory analysis results.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding for visualization
    result : TrajectoryResult
        Trajectory results
    labels : np.ndarray, optional
        Cluster labels for coloring
    show_arrows : bool
        Show trajectory direction arrows
    arrow_density : float
        Fraction of cells to show arrows for
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    n_plots = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot 1: Pseudotime
    ax = axes[0]
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=result.pseudotime, cmap='viridis',
        s=10, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label='Pseudotime')

    # Mark root cell
    ax.scatter(
        embedding[result.root_cell, 0],
        embedding[result.root_cell, 1],
        c='red', s=200, marker='*', edgecolors='black',
        label='Root', zorder=5
    )

    # Add arrows showing direction
    if show_arrows:
        n_arrows = int(len(embedding) * arrow_density)
        arrow_idx = np.random.choice(len(embedding), n_arrows, replace=False)

        for idx in arrow_idx:
            # Find neighbor with higher pseudotime
            dists = np.linalg.norm(embedding - embedding[idx], axis=1)
            neighbors = np.argsort(dists)[1:11]  # 10 nearest
            higher_pt = neighbors[result.pseudotime[neighbors] > result.pseudotime[idx]]

            if len(higher_pt) > 0:
                target = higher_pt[0]
                dx = embedding[target, 0] - embedding[idx, 0]
                dy = embedding[target, 1] - embedding[idx, 1]
                ax.arrow(
                    embedding[idx, 0], embedding[idx, 1],
                    dx * 0.3, dy * 0.3,
                    head_width=0.1, head_length=0.05,
                    fc='gray', ec='gray', alpha=0.5
                )

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Pseudotime')
    ax.legend()

    # Plot 2: Clusters (if provided)
    if labels is not None:
        ax = axes[1]
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                label=f'Cluster {label}', s=10, alpha=0.7
            )

        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Clusters')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_pseudotime_heatmap(
    expression: np.ndarray,
    pseudotime: np.ndarray,
    gene_names: np.ndarray,
    genes: Optional[List[str]] = None,
    n_genes: int = 50,
    n_bins: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot gene expression heatmap ordered by pseudotime.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    pseudotime : np.ndarray
        Pseudotime values
    gene_names : np.ndarray
        Gene names
    genes : List[str], optional
        Specific genes to plot
    n_genes : int
        Number of variable genes if genes not specified
    n_bins : int
        Number of pseudotime bins
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

    # Select genes
    if genes is not None:
        gene_mask = np.isin(gene_names, genes)
        selected_genes = gene_names[gene_mask]
        expr = expression[:, gene_mask]
    else:
        # Select most variable genes
        gene_var = np.var(expression, axis=0)
        top_idx = np.argsort(gene_var)[-n_genes:]
        selected_genes = gene_names[top_idx]
        expr = expression[:, top_idx]

    # Bin cells by pseudotime
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pseudotime, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Average expression per bin
    binned_expr = np.zeros((n_bins, expr.shape[1]))
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            binned_expr[i] = expr[mask].mean(axis=0)

    # Z-score normalize
    binned_expr = (binned_expr - binned_expr.mean(axis=0)) / (binned_expr.std(axis=0) + 1e-10)

    # Cluster genes by expression pattern
    from scipy.cluster.hierarchy import linkage, leaves_list
    if len(selected_genes) > 1:
        Z = linkage(binned_expr.T, method='ward')
        gene_order = leaves_list(Z)
    else:
        gene_order = np.arange(len(selected_genes))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        binned_expr[:, gene_order].T,
        aspect='auto', cmap='RdBu_r',
        vmin=-2, vmax=2
    )

    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Gene')
    ax.set_xticks([0, n_bins // 2, n_bins - 1])
    ax.set_xticklabels(['0', '0.5', '1'])

    if len(selected_genes) <= 30:
        ax.set_yticks(range(len(selected_genes)))
        ax.set_yticklabels(selected_genes[gene_order])
    else:
        ax.set_yticks([])

    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_title('Gene Expression Along Pseudotime')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def find_trajectory_genes(
    expression: np.ndarray,
    pseudotime: np.ndarray,
    gene_names: np.ndarray,
    n_genes: int = 100,
    method: str = 'correlation'
) -> pd.DataFrame:
    """
    Find genes that vary along pseudotime.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    pseudotime : np.ndarray
        Pseudotime values
    gene_names : np.ndarray
        Gene names
    n_genes : int
        Number of top genes to return
    method : str
        Method: 'correlation', 'variance'

    Returns
    -------
    pd.DataFrame
        Trajectory-associated genes
    """
    from scipy import stats

    n_genes_total = expression.shape[1]
    results = []

    for i in range(n_genes_total):
        gene_expr = expression[:, i]

        if method == 'correlation':
            corr, pval = stats.spearmanr(pseudotime, gene_expr)
            score = abs(corr)
        elif method == 'variance':
            # Variance explained by pseudotime bins
            bins = pd.cut(pseudotime, bins=10, labels=False)
            between_var = np.var([gene_expr[bins == b].mean() for b in range(10)])
            total_var = np.var(gene_expr)
            score = between_var / (total_var + 1e-10)
            corr = stats.spearmanr(pseudotime, gene_expr)[0]
            pval = 0
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            'gene': gene_names[i],
            'correlation': corr,
            'score': score,
            'pvalue': pval
        })

    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False).head(n_genes)

    return df
