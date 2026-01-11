from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from featurize import smiles_to_pyg
from model import PIGNN


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def standardize(x: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    return (x - mean) / scale


@torch.no_grad()
def main() -> None:
    out_dir = Path("outputs_model1")
    meta = load_json(out_dir / "meta.json")
    scalers = load_json(out_dir / "scalers.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PIGNN(
        node_dim=meta["node_dim"],
        phys_dim=meta["phys_dim"],
        gnn_hidden=meta["gnn_hidden"],
        gnn_layers=meta["gnn_layers"],
        mlp_hidden=meta["mlp_hidden"],
        z_dim=meta["z_dim"],
        k_theta=meta["k_theta"],
        dropout=meta["dropout"],
        pool=meta["pool"],
    ).to(device)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- Example input ----
    smiles = "CCO"
    phys = {
        "mw": 46.07,
        "vapor_pressure_pa": 5900,
        "boiling_point_k": 351.5,
    }

    # Ensure same order as training schema
    scalar_cols = meta["scalar_cols"]
    u = np.array([phys[c] for c in scalar_cols], dtype=np.float32)
    u_std = standardize(u, scalers["u_mean"], scalers["u_scale"])

    graph, _ = smiles_to_pyg(smiles)
    graph.u = torch.tensor(u_std, dtype=torch.float)
    # For single example, need batch vector of zeros
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    graph = graph.to(device)
    out = model(graph)

    # De-standardize theta back to original units
    theta_std = out.theta_pred.detach().cpu().numpy().reshape(-1)
    theta = theta_std * np.array(scalers["y_scale"], dtype=np.float32) + np.array(scalers["y_mean"], dtype=np.float32)

    z_chem = out.z_chem.detach().cpu().numpy().reshape(-1)

    print("Predicted theta:", theta)
    print("z_chem shape:", z_chem.shape)


if __name__ == "__main__":
    main()
