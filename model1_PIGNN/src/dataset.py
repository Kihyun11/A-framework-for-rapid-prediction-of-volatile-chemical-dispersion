from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset

from featurize import smiles_to_pyg


@dataclass
class Schema:
    id_col: str
    smiles_col: str
    scalar_cols: List[str]
    theta_cols: List[str]


def read_split_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


class ChemThetaDataset(Dataset):
    """
    Returns PyG Data objects with:
      - x, edge_index, edge_attr
      - u: [phys_dim] float tensor (graph-level scalar features)
      - y: [k_theta] float tensor (theta targets)
      - chem_id: stored as Python string attribute
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        schema: Schema,
        ids: List[str],
        scaler_u: Optional[StandardScaler] = None,
        scaler_y: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.schema = schema
        self.ids = ids

        # Join master + targets
        merged = master_df.merge(targets_df, on=schema.id_col, how="inner")
        merged = merged[merged[schema.id_col].isin(ids)].reset_index(drop=True)
        if len(merged) == 0:
            raise ValueError("No rows after join/filter. Check chem_id overlaps and split files.")

        self.df = merged

        # Fit or use scalers
        u_np = self.df[schema.scalar_cols].to_numpy(dtype=np.float32)
        y_np = self.df[schema.theta_cols].to_numpy(dtype=np.float32)

        if scaler_u is None:
            scaler_u = StandardScaler().fit(u_np)
        if scaler_y is None:
            scaler_y = StandardScaler().fit(y_np)

        self.scaler_u = scaler_u
        self.scaler_y = scaler_y

        self.u = scaler_u.transform(u_np).astype(np.float32)
        self.y = scaler_y.transform(y_np).astype(np.float32)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        chem_id = row[self.schema.id_col]
        smiles = row[self.schema.smiles_col]

        graph, _ = smiles_to_pyg(smiles)

        u = torch.tensor(self.u[idx], dtype=torch.float).unsqueeze(0)  # [phys_dim]
        y = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)  # [k_theta]

        graph.u = u
        graph.y = y
        graph.chem_id = chem_id
        return graph


def load_tables(
    data_dir: Path,
    master_csv: str,
    targets_csv: str,
    id_col: str,
    smiles_col: str,
    theta_prefix: str = "theta_",
    scalar_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Schema]:
    master_path = data_dir / master_csv
    targets_path = data_dir / targets_csv

    master_df = pd.read_csv(master_path)
    targets_df = pd.read_csv(targets_path)

    if id_col not in master_df.columns or id_col not in targets_df.columns:
        raise ValueError(f"Both tables must contain '{id_col}'.")

    if smiles_col not in master_df.columns:
        raise ValueError(f"master table must contain '{smiles_col}'.")

    theta_cols = [c for c in targets_df.columns if c.startswith(theta_prefix)]
    if len(theta_cols) == 0:
        raise ValueError(f"No theta columns found with prefix '{theta_prefix}' in targets table.")

    if scalar_cols is None:
        scalar_cols = [c for c in master_df.columns if c not in {id_col, smiles_col}]
    if len(scalar_cols) == 0:
        raise ValueError("No scalar columns found. Provide scalar_cols or add physchem columns to chem_master.csv.")

    schema = Schema(
        id_col=id_col,
        smiles_col=smiles_col,
        scalar_cols=scalar_cols,
        theta_cols=theta_cols,
    )
    return master_df, targets_df, schema
