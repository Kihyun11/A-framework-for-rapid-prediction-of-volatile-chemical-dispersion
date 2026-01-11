# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool


@dataclass
class ModelOutput:
    theta_pred: torch.Tensor  # [B, k_theta]
    z_chem: torch.Tensor      # [B, z_dim]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        dropout: float = 0.0,
        act: nn.Module = nn.SiLU(),
        layer_norm: bool = True,
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PIGNN(nn.Module):
    """
    Physicochemical Property–Based Dispersion Model (PI-GNN)

    Expects a PyG batch with:
      - data.x:         [N, node_dim]      atom/node features
      - data.edge_index:[2, E]            connectivity
      - data.batch:     [N]               node -> graph id
      - data.u:         [B, phys_dim]      scalar physicochemical properties (graph-level)

    Outputs:
      - theta_pred:     [B, k_theta]      supervised dispersion characteristics (targets)
      - z_chem:         [B, z_dim]        embedding for downstream fusion (Model ④)
    """

    def __init__(
        self,
        node_dim: int,
        phys_dim: int,
        gnn_hidden: int = 128,
        gnn_layers: int = 5,
        mlp_hidden: int = 256,
        z_dim: int = 128,
        k_theta: int = 10,
        dropout: float = 0.1,
        pool: str = "mean",  # "mean" | "add" | "max"
    ):
        super().__init__()

        if pool not in {"mean", "add", "max"}:
            raise ValueError(f"Invalid pool={pool}. Choose from mean/add/max.")

        self.pool = pool

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
            nn.SiLU(),
        )

        # GIN message passing blocks
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(gnn_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(gnn_hidden, gnn_hidden),
                nn.SiLU(),
                nn.Linear(gnn_hidden, gnn_hidden),
            )
            self.convs.append(GINConv(nn_mlp))
            self.bns.append(nn.BatchNorm1d(gnn_hidden))

        # Scalar physchem encoder
        self.phys_encoder = nn.Sequential(
            nn.Linear(phys_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Fusion MLP → shared representation
        fusion_in = gnn_hidden + gnn_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
        )

        # Two heads
        self.theta_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, k_theta),
        )
        self.z_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, z_dim),
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        if self.pool == "add":
            return global_add_pool(x, batch)
        return global_max_pool(x, batch)

    def forward(self, data: Data) -> ModelOutput:
        if not hasattr(data, "u"):
            raise ValueError("Input Data must contain graph-level scalar features `data.u` of shape [B, phys_dim].")

        x = self.node_encoder(data.x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.silu(x)

        h_graph = self._pool(x, data.batch)          # [B, gnn_hidden]
        h_phys = self.phys_encoder(data.u)           # [B, gnn_hidden]

        h = torch.cat([h_graph, h_phys], dim=-1)     # [B, 2*gnn_hidden]
        h = self.fusion(h)                           # [B, mlp_hidden]

        theta_pred = self.theta_head(h)              # [B, k_theta]
        z_chem = self.z_head(h)                      # [B, z_dim]
        return ModelOutput(theta_pred=theta_pred, z_chem=z_chem)
