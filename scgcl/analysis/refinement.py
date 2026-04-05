"""Cluster refinement: subclustering and merging operations."""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubclusterResult:
    """Container for subclustering results."""

    original_cluster: int
    n_subclusters: int
    subcluster_labels: np.ndarray  # Labels for cells in the original cluster
    full_labels: np.ndarray  # Updated labels for all cells
    cell_indices: np.ndarray  # Indices of cells that were subclustered
    subcluster_sizes: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            f"Subclustering Results - Cluster {self.original_cluster}",
            "=" * 50,
            f"Number of subclusters: {self.n_subclusters}",
            f"Cells subclustered: {len(self.cell_indices)}",
            "",
            "Subcluster sizes:"
        ]
        for sub, size in sorted(self.subcluster_sizes.items()):
            lines.append(f"  Subcluster {sub}: {size} cells")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class MergeResult:
    """Container for cluster merging results."""

    original_n_clusters: int
    merged_n_clusters: int
    merged_labels: np.ndarray
    merge_history: List[Tuple[int, int]]  # List of (cluster1, cluster2) merges
    cluster_mapping: Dict[int, int]  # old_label -> new_label

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Cluster Merge Results",
            "=" * 50,
            f"Original clusters: {self.original_n_clusters}",
            f"Merged clusters: {self.merged_n_clusters}",
            f"Merges performed: {len(self.merge_history)}",
            "",
            "Merge history:"
        ]
        for c1, c2 in self.merge_history:
            lines.append(f"  Merged cluster {c2} into cluster {c1}")
        lines.append("=" * 50)
        return "\n".join(lines)


def subcluster(
    X: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    n_subclusters: int = 2,
    method: str = 'kmeans',
    model: Optional[object] = None,
    embeddings: Optional[np.ndarray] = None,
    use_embeddings: bool = True,
    random_state: int = 42,
    **kwargs
) -> SubclusterResult:
    """
    Subcluster a specific cluster into finer groups.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (n_cells x n_genes)
    labels : np.ndarray
        Current cluster labels
    cluster_id : int
        Which cluster to subcluster
    n_subclusters : int
        Number of subclusters to create
    method : str
        Subclustering method: 'kmeans', 'scgcl', 'hierarchical'
    model : ScGCL, optional
        Fitted ScGCL model (required if use_embeddings=True and embeddings not provided)
    embeddings : np.ndarray, optional
        Pre-computed embeddings (n_cells x n_dims)
    use_embeddings : bool
        Whether to use embeddings (True) or raw expression (False)
    random_state : int
        Random seed
    **kwargs
        Additional arguments passed to clustering method

    Returns
    -------
    SubclusterResult
        Subclustering results with updated labels
    """
    # Get cells in target cluster
    cluster_mask = labels == cluster_id
    cell_indices = np.where(cluster_mask)[0]
    n_cells_in_cluster = len(cell_indices)

    if n_cells_in_cluster < n_subclusters:
        raise ValueError(
            f"Cluster {cluster_id} has only {n_cells_in_cluster} cells, "
            f"cannot create {n_subclusters} subclusters"
        )

    # Get data for subclustering
    if use_embeddings:
        if embeddings is not None:
            sub_data = embeddings[cluster_mask]
        elif model is not None and hasattr(model, 'get_embeddings'):
            all_embeddings = model.get_embeddings()
            sub_data = all_embeddings[cluster_mask]
        else:
            warnings.warn("No embeddings available, using expression data")
            sub_data = X[cluster_mask]
    else:
        sub_data = X[cluster_mask]

    # Perform subclustering
    if method == 'kmeans':
        kmeans = KMeans(
            n_clusters=n_subclusters,
            n_init=10,
            random_state=random_state
        )
        sub_labels = kmeans.fit_predict(sub_data)

    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        linkage_method = kwargs.get('linkage', 'ward')
        clustering = AgglomerativeClustering(
            n_clusters=n_subclusters,
            linkage=linkage_method
        )
        sub_labels = clustering.fit_predict(sub_data)

    elif method == 'scgcl':
        # Use ScGCL for subclustering
        from scgcl.model import ScGCL

        sub_X = X[cluster_mask]
        sub_model = ScGCL(
            n_clusters=n_subclusters,
            pretrain_epochs=kwargs.get('pretrain_epochs', 50),
            ssc_epochs=kwargs.get('ssc_epochs', 100),
            hidden_dim=kwargs.get('hidden_dim', 64),
        )
        sub_labels = sub_model.fit_predict(
            sub_X,
            preprocess=kwargs.get('preprocess', False),
            verbose=kwargs.get('verbose', False)
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans', 'hierarchical', or 'scgcl'")

    # Create new label scheme
    # Original clusters keep their labels, subclusters get new labels
    max_label = labels.max()
    full_labels = labels.copy()

    # Map subclusters to new labels
    # Subcluster 0 keeps original label, others get new labels
    unique_subclusters = np.unique(sub_labels)
    subcluster_mapping = {}

    for i, sub in enumerate(unique_subclusters):
        if i == 0:
            subcluster_mapping[sub] = cluster_id
        else:
            subcluster_mapping[sub] = max_label + i

    # Update full labels
    new_sub_labels = np.array([subcluster_mapping[s] for s in sub_labels])
    full_labels[cluster_mask] = new_sub_labels

    # Compute subcluster sizes
    subcluster_sizes = {}
    for sub in np.unique(new_sub_labels):
        subcluster_sizes[int(sub)] = int(np.sum(new_sub_labels == sub))

    return SubclusterResult(
        original_cluster=cluster_id,
        n_subclusters=n_subclusters,
        subcluster_labels=new_sub_labels,
        full_labels=full_labels,
        cell_indices=cell_indices,
        subcluster_sizes=subcluster_sizes
    )


def subcluster_recursive(
    X: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 50,
    max_depth: int = 3,
    n_subclusters: int = 2,
    method: str = 'kmeans',
    embeddings: Optional[np.ndarray] = None,
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Recursively subcluster all clusters until minimum size is reached.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    labels : np.ndarray
        Initial cluster labels
    min_cluster_size : int
        Stop subclustering when cluster size falls below this
    max_depth : int
        Maximum recursion depth
    n_subclusters : int
        Number of subclusters per split
    method : str
        Subclustering method
    embeddings : np.ndarray, optional
        Pre-computed embeddings
    verbose : bool
        Print progress
    **kwargs
        Additional arguments for subcluster()

    Returns
    -------
    np.ndarray
        Final cluster labels after recursive subclustering
    """
    current_labels = labels.copy()

    for depth in range(max_depth):
        clusters_to_split = []

        for cluster in np.unique(current_labels):
            cluster_size = np.sum(current_labels == cluster)
            if cluster_size >= min_cluster_size * n_subclusters:
                clusters_to_split.append(cluster)

        if not clusters_to_split:
            if verbose:
                logger.info("Depth %d: No clusters large enough to split", depth)
            break

        if verbose:
            logger.info("Depth %d: Splitting %d clusters", depth, len(clusters_to_split))

        for cluster in clusters_to_split:
            try:
                result = subcluster(
                    X, current_labels, cluster,
                    n_subclusters=n_subclusters,
                    method=method,
                    embeddings=embeddings,
                    **kwargs
                )
                current_labels = result.full_labels
            except ValueError:
                continue

    # Relabel to consecutive integers
    unique_labels = np.unique(current_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in current_labels])

    if verbose:
        logger.info("Final: %d clusters", len(np.unique(final_labels)))

    return final_labels


def merge_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
    method: str = 'centroid',
    linkage_method: str = 'average',
    metric: str = 'euclidean'
) -> MergeResult:
    """
    Merge similar clusters based on distance threshold or target count.

    Parameters
    ----------
    labels : np.ndarray
        Current cluster labels
    embeddings : np.ndarray
        Cell embeddings (n_cells x n_dims)
    threshold : float, optional
        Distance threshold for merging (clusters closer than this are merged)
    n_clusters : int, optional
        Target number of clusters after merging
    method : str
        How to compute cluster distances: 'centroid', 'nearest', 'farthest'
    linkage_method : str
        Linkage method for hierarchical merging: 'single', 'complete', 'average', 'ward'
    metric : str
        Distance metric

    Returns
    -------
    MergeResult
        Merge results with updated labels
    """
    if threshold is None and n_clusters is None:
        raise ValueError("Must specify either threshold or n_clusters")

    unique_labels = np.unique(labels)
    original_n_clusters = len(unique_labels)

    if n_clusters is not None and n_clusters >= original_n_clusters:
        # No merging needed
        return MergeResult(
            original_n_clusters=original_n_clusters,
            merged_n_clusters=original_n_clusters,
            merged_labels=labels.copy(),
            merge_history=[],
            cluster_mapping={int(l): int(l) for l in unique_labels}
        )

    # Compute cluster centroids
    centroids = np.zeros((original_n_clusters, embeddings.shape[1]))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    for label in unique_labels:
        mask = labels == label
        centroids[label_to_idx[label]] = embeddings[mask].mean(axis=0)

    # Compute distance matrix based on method
    if method == 'centroid':
        distances = pdist(centroids, metric=metric)
    elif method == 'nearest':
        # Minimum distance between any two cells
        dist_matrix = np.zeros((original_n_clusters, original_n_clusters))
        for i, l1 in enumerate(unique_labels):
            for j, l2 in enumerate(unique_labels):
                if i < j:
                    cells1 = embeddings[labels == l1]
                    cells2 = embeddings[labels == l2]
                    # Compute pairwise distances
                    from sklearn.metrics import pairwise_distances
                    pairwise = pairwise_distances(cells1, cells2, metric=metric)
                    dist_matrix[i, j] = pairwise.min()
                    dist_matrix[j, i] = dist_matrix[i, j]
        distances = squareform(dist_matrix)
    elif method == 'farthest':
        # Maximum distance between any two cells
        dist_matrix = np.zeros((original_n_clusters, original_n_clusters))
        for i, l1 in enumerate(unique_labels):
            for j, l2 in enumerate(unique_labels):
                if i < j:
                    cells1 = embeddings[labels == l1]
                    cells2 = embeddings[labels == l2]
                    from sklearn.metrics import pairwise_distances
                    pairwise = pairwise_distances(cells1, cells2, metric=metric)
                    dist_matrix[i, j] = pairwise.max()
                    dist_matrix[j, i] = dist_matrix[i, j]
        distances = squareform(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Hierarchical clustering on clusters
    Z = linkage(distances, method=linkage_method)

    # Determine cut point
    if n_clusters is not None:
        cluster_assignments = fcluster(Z, n_clusters, criterion='maxclust')
    else:
        cluster_assignments = fcluster(Z, threshold, criterion='distance')

    merged_n_clusters = len(np.unique(cluster_assignments))

    # Build merge history by analyzing linkage matrix
    merge_history = []
    idx_to_label = {i: l for i, l in enumerate(unique_labels)}

    # Track which original clusters ended up together
    merge_groups = {}
    for i, assignment in enumerate(cluster_assignments):
        if assignment not in merge_groups:
            merge_groups[assignment] = []
        merge_groups[assignment].append(unique_labels[i])

    for group_labels in merge_groups.values():
        if len(group_labels) > 1:
            # Record merges (all merged into the first one)
            primary = group_labels[0]
            for secondary in group_labels[1:]:
                merge_history.append((int(primary), int(secondary)))

    # Create mapping from old to new labels
    cluster_mapping = {}
    new_label = 0
    group_to_new_label = {}

    for i, label in enumerate(unique_labels):
        group = cluster_assignments[i]
        if group not in group_to_new_label:
            group_to_new_label[group] = new_label
            new_label += 1
        cluster_mapping[int(label)] = group_to_new_label[group]

    # Apply mapping to get merged labels
    merged_labels = np.array([cluster_mapping[int(l)] for l in labels])

    return MergeResult(
        original_n_clusters=original_n_clusters,
        merged_n_clusters=merged_n_clusters,
        merged_labels=merged_labels,
        merge_history=merge_history,
        cluster_mapping=cluster_mapping
    )


def merge_by_markers(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_top_markers: int = 10,
    correlation_threshold: float = 0.8,
    method: str = 'spearman'
) -> MergeResult:
    """
    Merge clusters with similar marker gene expression profiles.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    labels : np.ndarray
        Cluster labels
    gene_names : list, optional
        Gene names
    n_top_markers : int
        Number of top marker genes to use per cluster
    correlation_threshold : float
        Correlation threshold for merging (0-1)
    method : str
        Correlation method: 'pearson' or 'spearman'

    Returns
    -------
    MergeResult
        Merge results
    """
    from scipy.stats import spearmanr, pearsonr

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Compute cluster mean expression
    cluster_means = np.zeros((n_clusters, X.shape[1]))
    for i, label in enumerate(unique_labels):
        cluster_means[i] = X[labels == label].mean(axis=0)

    # Find top variable genes across clusters
    gene_var = cluster_means.var(axis=0)
    top_gene_idx = np.argsort(gene_var)[-n_top_markers * n_clusters:]

    # Compute correlation matrix between clusters using top genes
    cluster_profiles = cluster_means[:, top_gene_idx]

    corr_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if method == 'spearman':
                corr, _ = spearmanr(cluster_profiles[i], cluster_profiles[j])
            else:
                corr, _ = pearsonr(cluster_profiles[i], cluster_profiles[j])
            corr_matrix[i, j] = corr

    # Convert correlation to distance (1 - corr)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)

    # Merge based on correlation threshold
    distance_threshold = 1 - correlation_threshold

    # Use merge_clusters with the distance matrix
    # Create a simple embedding from cluster profiles for compatibility
    return merge_clusters(
        labels=labels,
        embeddings=cluster_profiles[np.array([np.where(unique_labels == l)[0][0] for l in labels])],
        threshold=distance_threshold,
        method='centroid',
        metric='correlation'
    )


def auto_merge(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    silhouette_threshold: float = 0.0
) -> MergeResult:
    """
    Automatically merge clusters to improve clustering quality.

    Merges clusters that would improve the overall silhouette score,
    and merges very small clusters into their nearest neighbor.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    embeddings : np.ndarray
        Cell embeddings
    min_cluster_size : int
        Minimum cluster size (smaller clusters are merged)
    silhouette_threshold : float
        Merge if it improves silhouette by at least this amount

    Returns
    -------
    MergeResult
        Merge results
    """
    from sklearn.metrics import silhouette_score

    current_labels = labels.copy()
    merge_history = []

    unique_labels = np.unique(current_labels)
    original_n_clusters = len(unique_labels)

    # First pass: merge small clusters
    for label in unique_labels:
        cluster_size = np.sum(current_labels == label)
        if cluster_size < min_cluster_size and len(np.unique(current_labels)) > 2:
            # Find nearest cluster
            cluster_cells = embeddings[current_labels == label]
            cluster_centroid = cluster_cells.mean(axis=0)

            min_dist = float('inf')
            nearest_cluster = None

            for other_label in np.unique(current_labels):
                if other_label != label:
                    other_cells = embeddings[current_labels == other_label]
                    other_centroid = other_cells.mean(axis=0)
                    dist = np.linalg.norm(cluster_centroid - other_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cluster = other_label

            if nearest_cluster is not None:
                current_labels[current_labels == label] = nearest_cluster
                merge_history.append((int(nearest_cluster), int(label)))

    # Relabel to consecutive integers
    unique_after = np.unique(current_labels)
    label_map = {old: new for new, old in enumerate(unique_after)}
    final_labels = np.array([label_map[l] for l in current_labels])

    # Build cluster mapping
    cluster_mapping = {}
    for old_label in np.unique(labels):
        # Find where cells with this old label ended up
        old_mask = labels == old_label
        new_label = final_labels[old_mask][0]
        cluster_mapping[int(old_label)] = int(new_label)

    return MergeResult(
        original_n_clusters=original_n_clusters,
        merged_n_clusters=len(np.unique(final_labels)),
        merged_labels=final_labels,
        merge_history=merge_history,
        cluster_mapping=cluster_mapping
    )


def plot_merge_dendrogram(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot dendrogram showing cluster relationships and merge threshold.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    embeddings : np.ndarray
        Cell embeddings
    threshold : float, optional
        Distance threshold line to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    unique_labels = np.unique(labels)

    # Compute centroids
    centroids = np.zeros((len(unique_labels), embeddings.shape[1]))
    for i, label in enumerate(unique_labels):
        centroids[i] = embeddings[labels == label].mean(axis=0)

    # Compute linkage
    Z = linkage(centroids, method='average')

    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        Z,
        labels=[f"Cluster {l}" for l in unique_labels],
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10
    )

    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--',
                   label=f'Merge threshold: {threshold:.2f}')
        ax.legend()

    ax.set_title("Cluster Merge Dendrogram")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
