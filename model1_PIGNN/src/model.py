from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool


@dataclass
class ModelOutput:
    theta_pred: torch.Tensor  # [B, k_theta]
    z_chem: torch.Tensor      # [B, z_dim]


class PIGNN(nn.Module):
    """
    Physics-Informed GNN for chemical -> theta prediction.

    Expects PyG Batch with:
      - x, edge_index, batch
      - u: graph-level scalar features
      - y: graph-level targets (used in train.py only)
    """

    def __init__(
        self,
        node_dim: int,
        phys_dim: int,
        gnn_hidden: int = 128,
        gnn_layers: int = 5,
        mlp_hidden: int = 256,
        z_dim: int = 128,
        k_theta: int = 1,
        dropout: float = 0.1,
        pool: str = "mean",  # mean/add/max
    ):
        super().__init__()

        if pool not in {"mean", "add", "max"}:
            raise ValueError(f"Invalid pool={pool}")

        self.pool = pool
        self.phys_dim = phys_dim

        # Encode node features to hidden dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
            nn.SiLU(),
        )

        # Message passing stack (GIN)
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

        # Encode scalar physchem features
        self.phys_encoder = nn.Sequential(
            nn.Linear(phys_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Fuse graph + scalar encodings
        fusion_in = gnn_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
        )

        # Heads
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

    def _normalize_u_shape(self, data: Data) -> torch.Tensor:
        """
        Make sure data.u becomes [B, phys_dim].

        Common cases:
          - [B, phys_dim]        -> OK
          - [B, 1, phys_dim]     -> squeeze(1)
          - [phys_dim]           -> [1, phys_dim]
          - [B*phys_dim]         -> ambiguous; likely PyG concatenation bug if stored 1D per sample
        """
        if not hasattr(data, "u"):
            raise ValueError("Batch must contain graph-level scalar features `data.u`.")

        u = data.u

        # Case: single graph inference might produce [phys_dim]
        if u.dim() == 1:
            if u.numel() == self.phys_dim:
                u = u.view(1, self.phys_dim)
            else:
                raise RuntimeError(
                    f"`data.u` is 1D with length {u.numel()}, expected phys_dim={self.phys_dim}. "
                    "Likely batching/featurization issue."
                )

        # Case: [B, 1, phys_dim]
        if u.dim() == 3 and u.size(1) == 1:
            u = u.squeeze(1)

        # Case: correct [B, phys_dim]
        if u.dim() == 2:
            if u.size(-1) != self.phys_dim:
                # Often happens when u got concatenated to [1, B*phys_dim]
                raise RuntimeError(
                    f"`data.u` has shape {tuple(u.shape)} but expected last dim phys_dim={self.phys_dim}. "
                    "This usually happens if you stored u as 1D per-sample and PyG concatenated it. "
                    "Fix: in dataset.py, set `graph.u = u.unsqueeze(0)` so each sample stores [1, phys_dim]."
                )
            return u

        raise RuntimeError(
            f"Unsupported `data.u` shape: {tuple(u.shape)}. "
            "Expected [B, phys_dim] (or [B,1,phys_dim]/[phys_dim])."
        )

    def forward(self, data: Data) -> ModelOutput:
        # Graph pathway
        x = self.node_encoder(data.x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.silu(x)

        h_graph = self._pool(x, data.batch)  # [B, gnn_hidden]

        # Scalar pathway
        u = self._normalize_u_shape(data)     # [B, phys_dim]
        h_phys = self.phys_encoder(u)         # [B, gnn_hidden]

        # Fusion
        h = torch.cat([h_graph, h_phys], dim=-1)
        h = self.fusion(h)

        theta_pred = self.theta_head(h)       # [B, k_theta]
        z_chem = self.z_head(h)               # [B, z_dim]
        return ModelOutput(theta_pred=theta_pred, z_chem=z_chem)
