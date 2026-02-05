"""Graph construction utilities."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def build_knn_graph(
    X: np.ndarray,
    k: int = 15,
    metric: str = 'euclidean',
    mode: str = 'connectivity'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-nearest neighbor graph.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples x n_features)
    k : int
        Number of neighbors
    metric : str
        Distance metric
    mode : str
        'connectivity' (binary) or 'distance' (weighted)

    Returns
    -------
    edge_index : torch.Tensor
        Edge indices [2, num_edges]
    edge_weight : torch.Tensor
        Edge weights
    """
    n_samples = X.shape[0]
    k_neighbors = min(k + 1, n_samples)

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric, algorithm='auto')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    row_indices, col_indices, weights = [], [], []

    for i in range(n_samples):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            w = 1.0 if mode == 'connectivity' else np.exp(-dist ** 2)
            weights.extend([w, w])

    return (torch.tensor([row_indices, col_indices], dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32))


def compute_snn_weights(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Compute Shared Nearest Neighbor edge weights (Jaccard similarity)."""
    neighbors = [set() for _ in range(n_nodes)]
    row, col = edge_index

    for i, j in zip(row.tolist(), col.tolist()):
        neighbors[i].add(j)
        neighbors[j].add(i)

    snn_weights = []
    for i, j in zip(row.tolist(), col.tolist()):
        n_i, n_j = neighbors[i], neighbors[j]
        jaccard = len(n_i & n_j) / max(len(n_i | n_j), 1)
        snn_weights.append(jaccard)

    weights = torch.tensor(snn_weights, dtype=torch.float32)
    return weights / weights.max() if weights.max() > 0 else weights


def build_adaptive_knn_graph(
    X: np.ndarray,
    k_min: int = 5,
    k_max: int = 30,
    local_scale: bool = True,
    metric: str = 'euclidean'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build adaptive kNN graph with density-based k selection.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    k_min : int
        Minimum neighbors
    k_max : int
        Maximum neighbors
    local_scale : bool
        Use local distance scaling
    """
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k_max + 1, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Local density
    local_density = 1.0 / (distances[:, 1:k_min+1].mean(axis=1) + 1e-8)
    local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)

    # Adaptive k: higher density -> lower k
    adaptive_k = k_min + ((1 - local_density) * (k_max - k_min)).astype(int)

    row_indices, col_indices, weights = [], [], []

    for i in range(n_samples):
        k_i = adaptive_k[i]
        for j_idx in range(1, k_i + 1):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            row_indices.append(i)
            col_indices.append(j)

            if local_scale:
                sigma = distances[i, k_min]
                weight = np.exp(-(dist ** 2) / (2 * sigma ** 2 + 1e-8))
            else:
                weight = np.exp(-dist ** 2)

            weights.append(weight)

    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Make symmetric
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight])

    return edge_index, edge_weight
