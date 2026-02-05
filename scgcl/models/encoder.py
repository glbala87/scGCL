"""Contrastive encoder with augmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .graph_conv import GraphConvEncoder
from .attention import GraphAttentionEncoder


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()

        layers = []
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()])

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()])

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ContrastiveEncoder(nn.Module):
    """
    Contrastive encoder with dual augmentation branches.

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden dimension for encoder
    proj_dim : int
        Projection head output dimension
    encoder_type : str
        Type of encoder ('gcn' or 'gat')
    num_layers : int
        Number of encoder layers
    num_heads : int
        Number of attention heads (for GAT)
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        proj_dim: int = 32,
        encoder_type: str = 'gcn',
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        if encoder_type == 'gcn':
            self.encoder = GraphConvEncoder(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                num_layers=num_layers, dropout=dropout
            )
        elif encoder_type == 'gat':
            self.encoder = GraphAttentionEncoder(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                num_layers=num_layers, num_heads=num_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.projector = ProjectionHead(hidden_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def encode_and_project(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.forward(x, edge_index, edge_weight)
        z = self.project(h)
        return h, z

    def contrastive_forward(
        self,
        x: torch.Tensor,
        edge_index_1: torch.Tensor,
        edge_index_2: torch.Tensor,
        edge_weight_1: Optional[torch.Tensor] = None,
        edge_weight_2: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None,
        x_2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning with two augmented views."""
        if x_1 is None:
            x_1 = x
        if x_2 is None:
            x_2 = x

        h = self.encoder(x, edge_index_1, edge_weight_1)
        h1 = self.encoder(x_1, edge_index_1, edge_weight_1)
        z1 = self.projector(h1)
        h2 = self.encoder(x_2, edge_index_2, edge_weight_2)
        z2 = self.projector(h2)

        return h, h1, h2, z1, z2
