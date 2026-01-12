from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    # Paths
    data_dir: Path = Path("data")
    master_csv: str = "chem_master.csv"
    targets_csv: str = "chem_targets_params.csv"
    splits_dir: str = "splits"

    # Columns
    id_col: str = "chem_id"
    smiles_col: str = "smiles"
    theta_prefix: str = "theta_"
    # If empty, we auto-detect scalar columns = all columns except id, smiles
    scalar_cols: List[str] | None = None

    # Model
    gnn_hidden: int = 128
    gnn_layers: int = 5
    mlp_hidden: int = 256
    z_dim: int = 128
    dropout: float = 0.1
    pool: str = "mean"  # mean/add/max

    # Training
    seed: int = 42
    batch_size: int = 32
    #original learning rate
    # lr: float = 3e-4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 12

    # Physics penalties (optional)
    lambda_nonneg: float = 0.1
    lambda_range: float = 0.0
    # Range penalty works if you provide per-theta bounds
    # (otherwise leave lambda_range=0)
