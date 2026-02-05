"""Graph Attention Network encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Network encoder with multi-head attention.

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output embedding dimension
    num_layers : int
        Number of GAT layers
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    attention_dropout : float
        Dropout on attention weights
    use_gatv2 : bool
        Use GATv2 (dynamic attention) instead of original GAT
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_gatv2: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        GATLayer = GATv2Conv if use_gatv2 else GATConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            GATLayer(input_dim, hidden_dim // num_heads, heads=num_heads,
                     dropout=attention_dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATLayer(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                         dropout=attention_dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        if num_layers > 1:
            self.convs.append(
                GATLayer(hidden_dim, output_dim, heads=1,
                         dropout=attention_dropout, concat=False)
            )
            self.norms.append(nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiScaleGATEncoder(nn.Module):
    """Multi-scale GAT capturing local and global structure."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4
    ):
        super().__init__()

        self.local_conv = GATv2Conv(input_dim, hidden_dim // 2, heads=num_heads, concat=False)
        self.global_proj = nn.Linear(input_dim, hidden_dim // 2)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        local_feat = F.elu(self.local_conv(x, edge_index))
        global_feat = self.global_proj(x.mean(dim=0, keepdim=True)).expand(x.size(0), -1)
        return self.fusion(torch.cat([local_feat, global_feat], dim=1))
