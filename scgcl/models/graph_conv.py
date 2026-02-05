"""Graph Convolutional Network encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GraphConvEncoder(nn.Module):
    """
    Multi-layer Graph Convolutional Network encoder.

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output embedding dimension
    num_layers : int
        Number of GCN layers
    activation : str
        Activation function ('relu', 'tanh', 'leaky_relu', 'elu', 'gelu')
    dropout : float
        Dropout probability
    batch_norm : bool
        Whether to use batch normalization
    residual : bool
        Whether to use residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = True,
        residual: bool = False
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        self.activation = activations.get(activation, nn.ReLU())

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        self.convs.append(GCNConv(input_dim, hidden_dim))
        if batch_norm:
            self.bns.append(BatchNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if batch_norm:
                self.bns.append(BatchNorm(hidden_dim))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
            if batch_norm:
                self.bns.append(BatchNorm(output_dim))

        if residual and input_dim != hidden_dim:
            self.res_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.res_proj = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index, edge_weight)

            if self.bns is not None:
                x = self.bns[i](x)

            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.residual and i > 0 and x_prev.shape == x.shape:
                    x = x + x_prev

        return x

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor,
                      edge_weight: torch.Tensor = None, normalize: bool = True) -> torch.Tensor:
        emb = self.forward(x, edge_index, edge_weight)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


class DeepGraphConvEncoder(nn.Module):
    """Deep GCN encoder with skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        return self.output_proj(x)
