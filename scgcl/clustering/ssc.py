"""Self-supervised clustering module."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List
from tqdm import tqdm

from ..losses.clustering import StudentTDistribution, compute_target_distribution, KLDivergenceLoss


class SelfSupervisedClustering(nn.Module):
    """Self-supervised clustering with learnable cluster centers."""

    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, embedding_dim))
        self.student_t = StudentTDistribution(alpha)
        self.kl_loss = KLDivergenceLoss()

    def initialize_centers(self, embeddings: torch.Tensor, method: str = 'kmeans', n_init: int = 10):
        embeddings_np = embeddings.detach().cpu().numpy()

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=42)
            kmeans.fit(embeddings_np)
            centers = kmeans.cluster_centers_
        else:
            idx = np.random.choice(len(embeddings_np), self.n_clusters, replace=False)
            centers = embeddings_np[idx]

        self.cluster_centers.data = torch.tensor(centers, dtype=embeddings.dtype, device=embeddings.device)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.student_t(embeddings, self.cluster_centers)
        return q, q.argmax(dim=1)

    def compute_loss(self, embeddings: torch.Tensor, target_p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.student_t(embeddings, self.cluster_centers)
        p = compute_target_distribution(q.detach()) if target_p is None else target_p
        return self.kl_loss(q, p), q

    def get_labels(self, embeddings: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            _, labels = self.forward(embeddings)
        return labels.cpu().numpy()


class ClusterRefiner:
    """Cluster refinement through self-supervised learning."""

    def __init__(
        self,
        n_clusters: int,
        max_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 0.01,
        momentum: float = 0.9,
        tol: float = 0.001,
        update_interval: int = 5,
        device: str = 'cpu'
    ):
        self.n_clusters = n_clusters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.tol = tol
        self.update_interval = update_interval
        self.device = device

    def refine(
        self,
        encoder: nn.Module,
        ssc: SelfSupervisedClustering,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[float]]:
        encoder.to(self.device)
        ssc.to(self.device)
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        optimizer = torch.optim.SGD(
            list(encoder.parameters()) + list(ssc.parameters()),
            lr=self.lr, momentum=self.momentum
        )

        n_samples = x.shape[0]
        losses = []
        prev_labels = None

        encoder.eval()
        with torch.no_grad():
            h = encoder(x, edge_index, edge_weight)
            q, _ = ssc(h)
            target_p = compute_target_distribution(q)

        iterator = tqdm(range(self.max_epochs), desc="SSC") if verbose else range(self.max_epochs)

        for epoch in iterator:
            encoder.train()
            ssc.train()

            indices = torch.randperm(n_samples, device=self.device)
            epoch_loss, n_batches = 0.0, 0

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                optimizer.zero_grad()

                h = encoder(x, edge_index, edge_weight)
                loss, _ = ssc.compute_loss(h[batch_idx], target_p[batch_idx])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / n_batches)

            if (epoch + 1) % self.update_interval == 0:
                encoder.eval()
                with torch.no_grad():
                    h = encoder(x, edge_index, edge_weight)
                    q, current_labels = ssc(h)
                    target_p = compute_target_distribution(q)

                    current_labels_np = current_labels.cpu().numpy()
                    if prev_labels is not None:
                        delta = np.sum(current_labels_np != prev_labels) / n_samples
                        if delta < self.tol:
                            if verbose:
                                tqdm.write(f"Converged at epoch {epoch+1}")
                            break
                    prev_labels = current_labels_np

        encoder.eval()
        with torch.no_grad():
            h = encoder(x, edge_index, edge_weight)
            _, final_labels = ssc(h)

        return final_labels.cpu().numpy(), losses


def tune_n_clusters(embeddings: np.ndarray, k_min: int = 2, k_max: int = 20, method: str = 'silhouette') -> Tuple[int, List[float]]:
    """Tune number of clusters using internal metrics."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    k_max = min(k_max, len(embeddings) - 1)
    scores = []

    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)

        if method == 'silhouette':
            scores.append(silhouette_score(embeddings, labels))
        elif method == 'calinski':
            scores.append(calinski_harabasz_score(embeddings, labels))
        elif method == 'davies':
            scores.append(-davies_bouldin_score(embeddings, labels))

    return k_min + np.argmax(scores), scores
