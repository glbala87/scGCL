"""Graph and feature augmentation for contrastive learning."""

import torch
from typing import Tuple, Optional


class GraphAugmentor:
    """Graph augmentation via edge/feature dropout."""

    def __init__(self, edge_drop_prob: float = 0.2, feature_drop_prob: float = 0.2):
        self.edge_drop_prob = edge_drop_prob
        self.feature_drop_prob = feature_drop_prob

    def augment(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        n_nodes: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_aug = self.augment_features(x)
        edge_index_aug, edge_weight_aug = self.augment_edges(edge_index, edge_weight)
        return x_aug, edge_index_aug, edge_weight_aug

    def augment_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_drop_prob == 0:
            return x
        mask = torch.rand(x.shape[1], device=x.device) > self.feature_drop_prob
        return x * mask.float()

    def augment_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.edge_drop_prob == 0:
            return edge_index, edge_weight

        n_edges = edge_index.shape[1]
        keep_mask = torch.rand(n_edges, device=edge_index.device) > self.edge_drop_prob
        edge_index = edge_index[:, keep_mask]

        if edge_weight is not None:
            edge_weight = edge_weight[keep_mask]

        return edge_index, edge_weight


class ContrastiveAugmentation:
    """Generate two differently augmented views for contrastive learning."""

    def __init__(
        self,
        edge_drop_prob: Tuple[float, float] = (0.2, 0.3),
        feature_drop_prob: Tuple[float, float] = (0.2, 0.3)
    ):
        self.aug_1 = GraphAugmentor(edge_drop_prob[0], feature_drop_prob[0])
        self.aug_2 = GraphAugmentor(edge_drop_prob[1], feature_drop_prob[1])

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple, Tuple]:
        view1 = self.aug_1.augment(x, edge_index, edge_weight)
        view2 = self.aug_2.augment(x, edge_index, edge_weight)
        return view1, view2
