from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from featurize import smiles_to_pyg
from model import PIGNN


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def standardize(x: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    return (x - mean) / scale


def destandardize(z: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    return z * scale + mean


def build_u_raw_with_g(
    smiles: str,
    scalar_cols: List[str],
    scalar_values: Dict[str, float],
) -> np.ndarray:
    """
    u_raw = [CSV scalars in scalar_cols order || g(LogP, TPSA, MolWt)]
    NOTE: g is computed from SMILES inside smiles_to_pyg().
    """
    # CSV scalars
    u_csv = np.array([float(scalar_values[c]) for c in scalar_cols], dtype=np.float32)

    # g from SMILES
    g_data, _ = smiles_to_pyg(smiles)
    g = g_data.g.detach().cpu().numpy().astype(np.float32)  # shape [3] expected

    u_raw = np.concatenate([u_csv, g], axis=0).astype(np.float32)
    return u_raw


def build_single_graph(
    smiles: str,
    scalar_cols: List[str],
    scalar_values: Dict[str, float],
    u_mean: List[float],
    u_scale: List[float],
) -> Data:
    """
    Build single PyG graph with standardized u (includes appended g).
    """
    g, _ = smiles_to_pyg(smiles)
    u_raw = build_u_raw_with_g(smiles, scalar_cols, scalar_values)
    u_std = standardize(u_raw, u_mean, u_scale)

    g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)  # [1, phys_dim]
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    return g


@torch.no_grad()
def predict_theta(
    model: PIGNN,
    g: Data,
    device: torch.device,
    y_mean: List[float],
    y_scale: List[float],
) -> np.ndarray:
    model.eval()
    g = g.to(device)
    out = model(g)
    theta_std = out.theta_pred.detach().cpu().numpy().reshape(-1)
    theta = destandardize(theta_std, y_mean, y_scale)
    return theta


def load_model_from_outputs(outputs_dir: Path, device: torch.device) -> Tuple[PIGNN, Dict, Dict]:
    meta = load_json(outputs_dir / "meta.json")
    scalers = load_json(outputs_dir / "scalers.json")

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

    ckpt = torch.load(outputs_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return model, meta, scalers


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    assert y_true.shape == y_pred.shape
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": float(r2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs_model1",
                        help="Directory containing best_model.pt, meta.json, scalers.json")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single")

    # single mode args
    parser.add_argument("--smiles", type=str, default="CCO", help="Unseen chemical SMILES")
    parser.add_argument("--scalar", type=str, nargs="*", default=[],
                        help='Scalar key-values like: T=298 (must match training scalar_cols)')

    # batch mode args
    parser.add_argument("--input_csv", type=str, default="data/zero_shot_input.csv",
                        help="CSV with columns: chem_id, smiles, <scalar_cols...> (unseen only)")
    parser.add_argument("--target_csv", type=str, default=None,
                        help="Optional CSV with columns: chem_id, theta_* (ground truth for metrics)")
    parser.add_argument("--id_col", type=str, default="chem_id")
    parser.add_argument("--smiles_col", type=str, default="smiles")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)

    model, meta, scalers = load_model_from_outputs(outputs_dir, device)

    scalar_cols: List[str] = meta["scalar_cols"]   # e.g., ["T"] now (M removed)
    theta_cols: List[str] = meta["theta_cols"]
    k_theta: int = int(meta["k_theta"])

    u_mean = scalers["u_mean"]
    u_scale = scalers["u_scale"]
    y_mean = scalers["y_mean"]
    y_scale = scalers["y_scale"]

    if args.mode == "single":
        scalar_values: Dict[str, float] = {}
        for kv in args.scalar:
            if "=" not in kv:
                raise ValueError(f"Bad --scalar entry: {kv}. Use key=value")
            k, v = kv.split("=", 1)
            scalar_values[k] = float(v)

        missing = [c for c in scalar_cols if c not in scalar_values]
        if missing:
            raise ValueError(
                f"Missing scalar inputs for columns: {missing}. "
                f"Provide them via --scalar like: {' '.join([f'{m}=...' for m in missing])}"
            )

        g = build_single_graph(args.smiles, scalar_cols, scalar_values, u_mean, u_scale)
        theta = predict_theta(model, g, device, y_mean, y_scale)

        print("=== ZERO-SHOT SINGLE PREDICTION ===")
        print("SMILES:", args.smiles)
        print("Scalars:", {k: scalar_values[k] for k in scalar_cols})
        print("Pred theta (original scale):", theta)
        print("Note: No exp/log conversion is applied automatically.")

    else:
        inp = pd.read_csv(args.input_csv)
        if args.id_col not in inp.columns or args.smiles_col not in inp.columns:
            raise ValueError(f"input_csv must include columns: {args.id_col}, {args.smiles_col}")

        for c in scalar_cols:
            if c not in inp.columns:
                raise ValueError(f"input_csv missing scalar column: {c}")

        graphs: List[Data] = []
        ids: List[str] = []

        for _, row in inp.iterrows():
            chem_id = str(row[args.id_col])
            smiles = str(row[args.smiles_col])
            scalar_values = {c: float(row[c]) for c in scalar_cols}

            g, _ = smiles_to_pyg(smiles)
            u_raw = build_u_raw_with_g(smiles, scalar_cols, scalar_values)
            u_std = standardize(u_raw, u_mean, u_scale)
            g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)

            g.chem_id = chem_id
            graphs.append(g)
            ids.append(chem_id)

        loader = DataLoader(graphs, batch_size=64, shuffle=False)

        preds_std = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                theta_std = out.theta_pred.detach().cpu().numpy()
                preds_std.append(theta_std)

        preds_std = np.concatenate(preds_std, axis=0)
        preds = destandardize(preds_std, y_mean, y_scale)

        out_df = pd.DataFrame({args.id_col: ids})
        for j in range(k_theta):
            out_df[f"pred_{theta_cols[j]}"] = preds[:, j]

        out_path = outputs_dir / "zero_shot_predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path}")

        if args.target_csv is not None:
            tgt = pd.read_csv(args.target_csv)
            if args.id_col not in tgt.columns:
                raise ValueError(f"target_csv must include column: {args.id_col}")
            tcols = [c for c in tgt.columns if c.startswith("theta_")]
            if len(tcols) != k_theta:
                raise ValueError(f"target_csv must have {k_theta} theta_* columns, found {tcols}")

            merged = out_df.merge(tgt[[args.id_col] + tcols], on=args.id_col, how="inner")
            if len(merged) == 0:
                raise ValueError("No matching chem_id between input_csv predictions and target_csv.")

            print("=== ZERO-SHOT METRICS ===")
            for j, col in enumerate(tcols):
                y_true = merged[col].to_numpy(dtype=np.float32)
                y_pred = merged[f"pred_{theta_cols[j]}"].to_numpy(dtype=np.float32)
                m = compute_metrics(y_true, y_pred)
                print(f"[{col}] MSE={m['mse']:.6f} RMSE={m['rmse']:.6f} MAE={m['mae']:.6f} R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()


#old ver 2
# from __future__ import annotations

# import argparse
# import json
# import math
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

# from featurize import smiles_to_pyg
# from model import PIGNN


# def load_json(path: Path) -> Dict:
#     return json.loads(path.read_text(encoding="utf-8"))


# def standardize(x: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
#     mean = np.array(mean, dtype=np.float32)
#     scale = np.array(scale, dtype=np.float32)
#     return (x - mean) / scale


# def destandardize(z: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
#     mean = np.array(mean, dtype=np.float32)
#     scale = np.array(scale, dtype=np.float32)
#     return z * scale + mean


# def build_u_raw_with_g(
#     smiles: str,
#     scalar_cols: List[str],
#     scalar_values: Dict[str, float],
# ) -> np.ndarray:
#     """
#     Build raw u vector in the SAME layout as training:
#       u_raw = [CSV scalars in scalar_cols order || g(LogP, TPSA)]
#     """
#     # CSV scalars
#     u_csv = np.array([float(scalar_values[c]) for c in scalar_cols], dtype=np.float32)

#     # g from SMILES (featurizer provides g = [LogP, TPSA])
#     g_data, _ = smiles_to_pyg(smiles)
#     g = g_data.g.detach().cpu().numpy().astype(np.float32)  # shape [2]

#     # concat
#     u_raw = np.concatenate([u_csv, g], axis=0).astype(np.float32)
#     return u_raw


# def build_single_graph(
#     smiles: str,
#     scalar_cols: List[str],
#     scalar_values: Dict[str, float],
#     u_mean: List[float],
#     u_scale: List[float],
# ) -> Data:
#     """
#     Build a single PyG Data graph with:
#       - x, edge_index, edge_attr
#       - u: [1, phys_dim] standardized (includes appended g)
#     """
#     g, _ = smiles_to_pyg(smiles)

#     u_raw = build_u_raw_with_g(smiles, scalar_cols, scalar_values)
#     u_std = standardize(u_raw, u_mean, u_scale)

#     g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)  # [1, phys_dim]
#     g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
#     return g


# @torch.no_grad()
# def predict_theta(
#     model: PIGNN,
#     g: Data,
#     device: torch.device,
#     y_mean: List[float],
#     y_scale: List[float],
# ) -> np.ndarray:
#     model.eval()
#     g = g.to(device)
#     out = model(g)
#     theta_std = out.theta_pred.detach().cpu().numpy().reshape(-1)  # [k_theta]
#     theta = destandardize(theta_std, y_mean, y_scale)
#     return theta


# def load_model_from_outputs(outputs_dir: Path, device: torch.device) -> Tuple[PIGNN, Dict, Dict]:
#     meta = load_json(outputs_dir / "meta.json")
#     scalers = load_json(outputs_dir / "scalers.json")

#     model = PIGNN(
#         node_dim=meta["node_dim"],
#         phys_dim=meta["phys_dim"],
#         gnn_hidden=meta["gnn_hidden"],
#         gnn_layers=meta["gnn_layers"],
#         mlp_hidden=meta["mlp_hidden"],
#         z_dim=meta["z_dim"],
#         k_theta=meta["k_theta"],
#         dropout=meta["dropout"],
#         pool=meta["pool"],
#     ).to(device)

#     ckpt = torch.load(outputs_dir / "best_model.pt", map_location=device)
#     model.load_state_dict(ckpt["state_dict"])
#     return model, meta, scalers


# def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     assert y_true.shape == y_pred.shape
#     mse = float(np.mean((y_true - y_pred) ** 2))
#     rmse = float(math.sqrt(mse))
#     mae = float(np.mean(np.abs(y_true - y_pred)))

#     ss_res = float(np.sum((y_true - y_pred) ** 2))
#     ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
#     r2 = 1.0 - ss_res / ss_tot
#     return {"mse": mse, "rmse": rmse, "mae": mae, "r2": float(r2)}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--outputs_dir", type=str, default="outputs_model1")
#     parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single")

#     # single mode
#     parser.add_argument("--smiles", type=str, default="CCO")
#     parser.add_argument("--scalar", type=str, nargs="*", default=[],
#                         help='Scalar key-values like: T=298 M=46.07 (must match training scalar_cols)')

#     # batch mode
#     parser.add_argument("--input_csv", type=str, default="data/zero_shot_input.csv")
#     parser.add_argument("--target_csv", type=str, default=None)
#     parser.add_argument("--id_col", type=str, default="chem_id")
#     parser.add_argument("--smiles_col", type=str, default="smiles")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     outputs_dir = Path(args.outputs_dir)

#     model, meta, scalers = load_model_from_outputs(outputs_dir, device)

#     scalar_cols: List[str] = meta["scalar_cols"]
#     theta_cols: List[str] = meta["theta_cols"]
#     k_theta: int = int(meta["k_theta"])

#     u_mean = scalers["u_mean"]
#     u_scale = scalers["u_scale"]
#     y_mean = scalers["y_mean"]
#     y_scale = scalers["y_scale"]

#     if args.mode == "single":
#         scalar_values: Dict[str, float] = {}
#         for kv in args.scalar:
#             if "=" not in kv:
#                 raise ValueError(f"Bad --scalar entry: {kv}. Use key=value")
#             k, v = kv.split("=", 1)
#             scalar_values[k] = float(v)

#         missing = [c for c in scalar_cols if c not in scalar_values]
#         if missing:
#             raise ValueError(
#                 f"Missing scalar inputs for columns: {missing}. "
#                 f"Provide them via --scalar like: {' '.join([f'{m}=...' for m in missing])}"
#             )

#         g = build_single_graph(args.smiles, scalar_cols, scalar_values, u_mean, u_scale)
#         theta = predict_theta(model, g, device, y_mean, y_scale)

#         print("=== ZERO-SHOT SINGLE PREDICTION ===")
#         print("SMILES:", args.smiles)
#         print("Scalars:", {k: scalar_values[k] for k in scalar_cols})
#         print("Pred theta (original scale):", theta)

#         # IMPORTANT:
#         # We DO NOT exp() here anymore.
#         # If your theta column is raw Psat, theta is raw Psat.
#         # If your theta column is log Psat, theta is log Psat. (You can manually exp if you want.)

#     else:
#         inp = pd.read_csv(args.input_csv)
#         if args.id_col not in inp.columns or args.smiles_col not in inp.columns:
#             raise ValueError(f"input_csv must include columns: {args.id_col}, {args.smiles_col}")

#         for c in scalar_cols:
#             if c not in inp.columns:
#                 raise ValueError(f"input_csv missing scalar column: {c}")

#         graphs: List[Data] = []
#         ids: List[str] = []

#         for _, row in inp.iterrows():
#             chem_id = str(row[args.id_col])
#             smiles = str(row[args.smiles_col])
#             scalar_values = {c: float(row[c]) for c in scalar_cols}

#             g, _ = smiles_to_pyg(smiles)

#             u_raw = build_u_raw_with_g(smiles, scalar_cols, scalar_values)
#             u_std = standardize(u_raw, u_mean, u_scale)
#             g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)

#             g.chem_id = chem_id
#             graphs.append(g)
#             ids.append(chem_id)

#         loader = DataLoader(graphs, batch_size=64, shuffle=False)

#         preds_std = []
#         model.eval()
#         with torch.no_grad():
#             for batch in loader:
#                 batch = batch.to(device)
#                 out = model(batch)
#                 theta_std = out.theta_pred.detach().cpu().numpy()
#                 preds_std.append(theta_std)

#         preds_std = np.concatenate(preds_std, axis=0)  # [N, k_theta]
#         preds = destandardize(preds_std, y_mean, y_scale)

#         out_df = pd.DataFrame({args.id_col: ids})
#         for j in range(k_theta):
#             out_df[f"pred_{theta_cols[j]}"] = preds[:, j]

#         out_path = outputs_dir / "zero_shot_predictions.csv"
#         out_df.to_csv(out_path, index=False)
#         print(f"Saved predictions to: {out_path}")

#         if args.target_csv is not None:
#             tgt = pd.read_csv(args.target_csv)
#             if args.id_col not in tgt.columns:
#                 raise ValueError(f"target_csv must include column: {args.id_col}")
#             tcols = [c for c in tgt.columns if c.startswith("theta_")]
#             if len(tcols) != k_theta:
#                 raise ValueError(f"target_csv must have {k_theta} theta_* columns, found {tcols}")

#             merged = out_df.merge(tgt[[args.id_col] + tcols], on=args.id_col, how="inner")
#             if len(merged) == 0:
#                 raise ValueError("No matching chem_id between input_csv predictions and target_csv.")

#             print("=== ZERO-SHOT METRICS ===")
#             for j, col in enumerate(tcols):
#                 y_true = merged[col].to_numpy(dtype=np.float32)
#                 y_pred = merged[f"pred_{theta_cols[j]}"].to_numpy(dtype=np.float32)
#                 m = compute_metrics(y_true, y_pred)
#                 print(f"[{col}] MSE={m['mse']:.6f} RMSE={m['rmse']:.6f} MAE={m['mae']:.6f} R2={m['r2']:.4f}")


# if __name__ == "__main__":
#     main()


#Old version
# from __future__ import annotations

# import argparse
# import json
# import math
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Batch, Data
# from torch_geometric.loader import DataLoader

# from featurize import smiles_to_pyg
# from model import PIGNN


# def load_json(path: Path) -> Dict:
#     return json.loads(path.read_text(encoding="utf-8"))


# def standardize(x: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
#     mean = np.array(mean, dtype=np.float32)
#     scale = np.array(scale, dtype=np.float32)
#     return (x - mean) / scale


# def destandardize(z: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
#     mean = np.array(mean, dtype=np.float32)
#     scale = np.array(scale, dtype=np.float32)
#     return z * scale + mean


# def build_single_graph(
#     smiles: str,
#     scalar_cols: List[str],
#     scalar_values: Dict[str, float],
#     u_mean: List[float],
#     u_scale: List[float],
# ) -> Data:
#     """
#     Build a single PyG Data graph with:
#       - x, edge_index, edge_attr
#       - u: [1, phys_dim] standardized
#       - batch: [num_nodes] zeros
#     """
#     g, _ = smiles_to_pyg(smiles)

#     # Ensure scalar order matches training
#     u_raw = np.array([float(scalar_values[c]) for c in scalar_cols], dtype=np.float32)
#     u_std = standardize(u_raw, u_mean, u_scale)

#     g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)  # [1, phys_dim]
#     g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
#     return g


# @torch.no_grad()
# def predict_theta(
#     model: PIGNN,
#     g: Data,
#     device: torch.device,
#     y_mean: List[float],
#     y_scale: List[float],
# ) -> np.ndarray:
#     """
#     Returns theta in original (de-standardized) scale: shape [k_theta]
#     """
#     model.eval()
#     g = g.to(device)
#     out = model(g)
#     theta_std = out.theta_pred.detach().cpu().numpy().reshape(-1)  # [k_theta]
#     theta = destandardize(theta_std, y_mean, y_scale)
#     return theta


# def load_model_from_outputs(outputs_dir: Path, device: torch.device) -> Tuple[PIGNN, Dict, Dict]:
#     meta = load_json(outputs_dir / "meta.json")
#     scalers = load_json(outputs_dir / "scalers.json")

#     model = PIGNN(
#         node_dim=meta["node_dim"],
#         phys_dim=meta["phys_dim"],
#         gnn_hidden=meta["gnn_hidden"],
#         gnn_layers=meta["gnn_layers"],
#         mlp_hidden=meta["mlp_hidden"],
#         z_dim=meta["z_dim"],
#         k_theta=meta["k_theta"],
#         dropout=meta["dropout"],
#         pool=meta["pool"],
#     ).to(device)

#     ckpt = torch.load(outputs_dir / "best_model.pt", map_location=device)
#     model.load_state_dict(ckpt["state_dict"])
#     return model, meta, scalers


# def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     """
#     y_true, y_pred: shape [N] for single-theta regression
#     """
#     assert y_true.shape == y_pred.shape
#     mse = float(np.mean((y_true - y_pred) ** 2))
#     rmse = float(math.sqrt(mse))
#     mae = float(np.mean(np.abs(y_true - y_pred)))

#     # R^2
#     ss_res = float(np.sum((y_true - y_pred) ** 2))
#     ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
#     r2 = 1.0 - ss_res / ss_tot
#     return {"mse": mse, "rmse": rmse, "mae": mae, "r2": float(r2)}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--outputs_dir", type=str, default="outputs_model1",
#                         help="Directory containing best_model.pt, meta.json, scalers.json")
#     parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single")

#     # single mode args
#     parser.add_argument("--smiles", type=str, default="CCO", help="Unseen chemical SMILES")
#     parser.add_argument("--scalar", type=str, nargs="*", default=[],
#                         help='Scalar key-values like: T=298 M=46.07 (must match training scalar_cols)')

#     # batch mode args
#     parser.add_argument("--input_csv", type=str, default="data/zero_shot_input.csv",
#                         help="CSV with columns: chem_id, smiles, <scalar_cols...> (unseen only)")
#     parser.add_argument("--target_csv", type=str, default=None,
#                         help="Optional CSV with columns: chem_id, theta_* (ground truth for metrics)")
#     parser.add_argument("--id_col", type=str, default="chem_id")
#     parser.add_argument("--smiles_col", type=str, default="smiles")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     outputs_dir = Path(args.outputs_dir)

#     model, meta, scalers = load_model_from_outputs(outputs_dir, device)

#     scalar_cols: List[str] = meta["scalar_cols"]
#     theta_cols: List[str] = meta["theta_cols"]  # names in training targets
#     k_theta: int = int(meta["k_theta"])

#     u_mean = scalers["u_mean"]
#     u_scale = scalers["u_scale"]
#     y_mean = scalers["y_mean"]
#     y_scale = scalers["y_scale"]

#     if args.mode == "single":
#         # Parse scalar key-values
#         scalar_values: Dict[str, float] = {}
#         for kv in args.scalar:
#             if "=" not in kv:
#                 raise ValueError(f"Bad --scalar entry: {kv}. Use key=value")
#             k, v = kv.split("=", 1)
#             scalar_values[k] = float(v)

#         # Ensure all needed scalars exist
#         missing = [c for c in scalar_cols if c not in scalar_values]
#         if missing:
#             raise ValueError(
#                 f"Missing scalar inputs for columns: {missing}. "
#                 f"Provide them via --scalar like: {' '.join([f'{m}=...' for m in missing])}"
#             )

#         g = build_single_graph(args.smiles, scalar_cols, scalar_values, u_mean, u_scale)
#         theta = predict_theta(model, g, device, y_mean, y_scale)

#         print("=== ZERO-SHOT SINGLE PREDICTION ===")
#         print("SMILES:", args.smiles)
#         print("Scalars:", {k: scalar_values[k] for k in scalar_cols})
#         print("Pred theta (original scale):", theta)

#         # If you trained on log(Psat), theta is log(Psat). Optionally convert:
#         if k_theta == 1 and ("log" in theta_cols[0].lower() or "log" in theta_cols[0].lower()):
#             psat = float(np.exp(theta[0]))
#             print("Derived Psat (exp(theta)):", psat)

#     else:
#         # Batch mode
#         inp = pd.read_csv(args.input_csv)
#         if args.id_col not in inp.columns or args.smiles_col not in inp.columns:
#             raise ValueError(f"input_csv must include columns: {args.id_col}, {args.smiles_col}")

#         for c in scalar_cols:
#             if c not in inp.columns:
#                 raise ValueError(f"input_csv missing scalar column: {c}")

#         # Build Data objects list
#         graphs: List[Data] = []
#         ids: List[str] = []
#         for _, row in inp.iterrows():
#             chem_id = str(row[args.id_col])
#             smiles = str(row[args.smiles_col])

#             scalar_values = {c: float(row[c]) for c in scalar_cols}
#             g, _ = smiles_to_pyg(smiles)

#             u_raw = np.array([scalar_values[c] for c in scalar_cols], dtype=np.float32)
#             u_std = standardize(u_raw, u_mean, u_scale)
#             g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)  # [1, phys_dim]

#             g.chem_id = chem_id
#             graphs.append(g)
#             ids.append(chem_id)

#         loader = DataLoader(graphs, batch_size=64, shuffle=False)

#         preds_std = []
#         model.eval()
#         with torch.no_grad():
#             for batch in loader:
#                 batch = batch.to(device)
#                 out = model(batch)
#                 theta_std = out.theta_pred.detach().cpu().numpy()  # [B, k_theta]
#                 preds_std.append(theta_std)

#         preds_std = np.concatenate(preds_std, axis=0)  # [N, k_theta]
#         preds = destandardize(preds_std, y_mean, y_scale)  # [N, k_theta]

#         # Save predictions
#         out_df = pd.DataFrame({args.id_col: ids})
#         for j in range(k_theta):
#             out_df[f"pred_{theta_cols[j]}"] = preds[:, j]

#         out_path = outputs_dir / "zero_shot_predictions.csv"
#         out_df.to_csv(out_path, index=False)
#         print(f"Saved predictions to: {out_path}")

#         # Optional metrics if target provided
#         if args.target_csv is not None:
#             tgt = pd.read_csv(args.target_csv)
#             if args.id_col not in tgt.columns:
#                 raise ValueError(f"target_csv must include column: {args.id_col}")
#             tcols = [c for c in tgt.columns if c.startswith("theta_")]
#             if len(tcols) != k_theta:
#                 raise ValueError(f"target_csv must have {k_theta} theta_* columns, found {tcols}")

#             merged = out_df.merge(tgt[[args.id_col] + tcols], on=args.id_col, how="inner")
#             if len(merged) == 0:
#                 raise ValueError("No matching chem_id between input_csv predictions and target_csv.")

#             # Metrics (supports k_theta==1 cleanly; multi-target -> report per-dim)
#             print("=== ZERO-SHOT METRICS ===")
#             for j, col in enumerate(tcols):
#                 y_true = merged[col].to_numpy(dtype=np.float32)
#                 y_pred = merged[f"pred_{theta_cols[j]}"].to_numpy(dtype=np.float32)
#                 m = compute_metrics(y_true, y_pred)
#                 print(f"[{col}] MSE={m['mse']:.6f} RMSE={m['rmse']:.6f} MAE={m['mae']:.6f} R2={m['r2']:.4f}")


# if __name__ == "__main__":
#     main()
