"""Enhanced visualization functions for cluster analysis."""

import numpy as np
from typing import Optional, List, Tuple, Union
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage
from scipy.spatial.distance import pdist


def silhouette_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'tab10',
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Create a silhouette plot showing per-cell silhouette scores.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings (n_cells x n_dims)
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure

    Returns
    -------
    silhouette_values : np.ndarray
        Per-cell silhouette scores
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from sklearn.metrics import silhouette_samples, silhouette_score
    except ImportError:
        print("matplotlib and sklearn required for plotting")
        return None

    # Compute silhouette scores
    silhouette_avg = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)

    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=figsize)

    y_lower = 10
    colormap = cm.get_cmap(cmap)

    for i, cluster in enumerate(sorted(np.unique(labels))):
        # Get silhouette values for cluster
        cluster_silhouette_values = sample_silhouette_values[labels == cluster]
        cluster_silhouette_values.sort()

        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size

        color = colormap(i / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7
        )

        # Label cluster
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster))

        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot (avg = {silhouette_avg:.3f})")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")

    # Add average line
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f"Average: {silhouette_avg:.3f}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.legend(loc="best")
    ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return sample_silhouette_values


def cluster_dendrogram(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'ward',
    metric: str = 'euclidean',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_leaf_labels: bool = True
) -> dict:
    """
    Create a dendrogram showing hierarchical cluster relationships.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings (n_cells x n_dims)
    labels : np.ndarray
        Cluster labels
    method : str
        Linkage method: 'ward', 'complete', 'average', 'single'
    metric : str
        Distance metric
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show_leaf_labels : bool
        Show cluster labels on leaves

    Returns
    -------
    dendrogram_info : dict
        Dendrogram information from scipy
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return {}

    # Compute cluster centroids
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    centroids = np.zeros((n_clusters, embeddings.shape[1]))
    for i, cluster in enumerate(unique_labels):
        centroids[i] = embeddings[labels == cluster].mean(axis=0)

    # Compute linkage
    if method == 'ward':
        Z = linkage(centroids, method='ward')
    else:
        distances = pdist(centroids, metric=metric)
        Z = linkage(distances, method=method)

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=figsize)

    if show_leaf_labels:
        leaf_labels = [f"Cluster {i}" for i in unique_labels]
    else:
        leaf_labels = None

    dend = scipy_dendrogram(
        Z,
        labels=leaf_labels,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10
    )

    ax.set_title("Cluster Dendrogram")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return dend


def cluster_heatmap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_genes: Optional[int] = None,
    gene_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> None:
    """
    Create a heatmap of cluster centroids.

    Parameters
    ----------
    embeddings : np.ndarray
        Expression matrix or embeddings
    labels : np.ndarray
        Cluster labels
    n_genes : int, optional
        Number of top variable genes to show
    gene_names : list, optional
        Gene names
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Compute cluster means
    cluster_means = np.zeros((n_clusters, embeddings.shape[1]))
    for i, cluster in enumerate(unique_labels):
        cluster_means[i] = embeddings[labels == cluster].mean(axis=0)

    # Select top variable features
    if n_genes is not None and n_genes < cluster_means.shape[1]:
        var = cluster_means.var(axis=0)
        top_idx = np.argsort(var)[-n_genes:]
        cluster_means = cluster_means[:, top_idx]
        if gene_names is not None:
            gene_names = [gene_names[i] for i in top_idx]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cluster_means,
        cmap=cmap,
        xticklabels=gene_names if gene_names else False,
        yticklabels=[f"Cluster {i}" for i in unique_labels],
        ax=ax
    )

    ax.set_title("Cluster Expression Heatmap")
    ax.set_xlabel("Features")
    ax.set_ylabel("Cluster")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_confidence_distribution(
    confidence: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot confidence score distribution per cluster.

    Parameters
    ----------
    confidence : np.ndarray
        Confidence scores
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("matplotlib, seaborn, and pandas required for plotting")
        return

    df = pd.DataFrame({
        'Confidence': confidence,
        'Cluster': labels.astype(str)
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Boxplot
    sns.boxplot(data=df, x='Cluster', y='Confidence', ax=axes[0])
    axes[0].set_title('Confidence by Cluster')

    # Violin plot
    sns.violinplot(data=df, x='Cluster', y='Confidence', ax=axes[1])
    axes[1].set_title('Confidence Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_cluster_composition(
    labels: np.ndarray,
    batch: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot cluster composition (size and optionally batch distribution).

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    batch : np.ndarray, optional
        Batch labels for composition analysis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib and pandas required for plotting")
        return

    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)

    if batch is None:
        # Simple cluster size plot
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))

        sizes = [np.sum(labels == c) for c in unique_labels]
        ax.bar(unique_labels, sizes, color='steelblue')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Cells')
        ax.set_title('Cluster Sizes')

    else:
        # Cluster composition by batch
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Cluster sizes
        sizes = [np.sum(labels == c) for c in unique_labels]
        axes[0].bar(unique_labels, sizes, color='steelblue')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Number of Cells')
        axes[0].set_title('Cluster Sizes')

        # Batch composition per cluster
        unique_batches = sorted(np.unique(batch))
        composition = np.zeros((n_clusters, len(unique_batches)))

        for i, c in enumerate(unique_labels):
            cluster_batch = batch[labels == c]
            for j, b in enumerate(unique_batches):
                composition[i, j] = np.sum(cluster_batch == b)

        # Normalize
        composition = composition / composition.sum(axis=1, keepdims=True)

        # Stacked bar
        bottom = np.zeros(n_clusters)
        for j, b in enumerate(unique_batches):
            axes[1].bar(unique_labels, composition[:, j], bottom=bottom, label=str(b))
            bottom += composition[:, j]

        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Proportion')
        axes[1].set_title('Batch Composition per Cluster')
        axes[1].legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
