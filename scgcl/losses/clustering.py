"""Clustering loss functions for self-supervised clustering."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for self-supervised clustering."""

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = q + 1e-10
        p = p + 1e-10
        return (p * torch.log(p / q)).sum(dim=1).mean()


class StudentTDistribution:
    """Student's t-distribution for soft cluster assignment."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, embeddings: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, cluster_centers, p=2).pow(2)
        q = (1 + dist / self.alpha).pow(-(self.alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)


def compute_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute target distribution P from soft assignments Q."""
    weight = q ** 2 / q.sum(dim=0, keepdim=True)
    return weight / weight.sum(dim=1, keepdim=True)


class ClusteringLoss(nn.Module):
    """Combined clustering loss for SSC refinement."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.student_t = StudentTDistribution(alpha)
        self.kl_loss = KLDivergenceLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        cluster_centers: torch.Tensor,
        target_p: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.student_t(embeddings, cluster_centers)
        p = compute_target_distribution(q.detach()) if target_p is None else target_p
        return self.kl_loss(q, p), q


class EntropyMinimizationLoss(nn.Module):
    """Entropy minimization for confident predictions."""

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        q = q + 1e-10
        return -(q * torch.log(q)).sum(dim=1).mean()


class ClusterBalanceLoss(nn.Module):
    """Loss to encourage balanced cluster sizes."""

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        cluster_dist = q.mean(dim=0)
        return (cluster_dist * torch.log(cluster_dist + 1e-10)).sum()
