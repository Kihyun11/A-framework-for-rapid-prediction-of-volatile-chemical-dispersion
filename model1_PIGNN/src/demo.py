from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch

from featurize import smiles_to_pyg
from model import PIGNN


# -----------------------
# Config (edit if needed)
# -----------------------
OUTPUTS_DIR = Path("outputs_model1")  # contains best_model.pt, meta.json, scalers.json


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def standardize(x: np.ndarray, mean, scale) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    return (x - mean) / scale


def destandardize(z: np.ndarray, mean, scale) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    return z * scale + mean


def load_model(outputs_dir: Path, device: torch.device):
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

    # ✅ silence PyTorch pickle warning + future-proof
    ckpt = torch.load(outputs_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, meta, scalers


@torch.no_grad()
def predict_single(model: PIGNN, meta: dict, scalers: dict, device: torch.device,
                   smiles: str, M: float, T: float):
    """
    Builds one graph, creates standardized u = [M, T, ... scalars || g(LogP,TPSA)],
    runs model, returns theta in original scale.
    """

    scalar_cols = meta["scalar_cols"]          # e.g., ["M","T"] or more
    theta_cols = meta["theta_cols"]            # e.g., ["theta_Psat"] or ["theta_log_Psat"]
    k_theta = int(meta["k_theta"])

    u_mean = scalers["u_mean"]
    u_scale = scalers["u_scale"]
    y_mean = scalers["y_mean"]
    y_scale = scalers["y_scale"]

    # Only M,T are in the UI; if training used more scalars, block with a friendly message.
    scalar_values = {}
    if "M" in scalar_cols:
        scalar_values["M"] = float(M)
    if "T" in scalar_cols:
        scalar_values["T"] = float(T)

    missing = [c for c in scalar_cols if c not in scalar_values]
    if missing:
        raise ValueError(
            f"Your model expects scalar columns {scalar_cols}, "
            f"but demo UI only provides M and T. Missing: {missing}. "
            f"Either (1) retrain with only M,T scalars, or (2) extend the UI."
        )

    # Build graph and g=[LogP,TPSA]
    g, _ = smiles_to_pyg(smiles)

    # u_raw = [CSV scalars || g]
    u_csv = np.array([float(scalar_values[c]) for c in scalar_cols], dtype=np.float32)
    g_vec = g.g.detach().cpu().numpy().astype(np.float32)  # [2] = [LogP, TPSA]
    u_raw = np.concatenate([u_csv, g_vec], axis=0)         # [phys_dim_total]

    # standardize u
    u_std = standardize(u_raw, u_mean, u_scale)
    g.u = torch.tensor(u_std, dtype=torch.float32).unsqueeze(0)  # [1, phys_dim]
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

    g = g.to(device)
    out = model(g)

    theta_std = out.theta_pred.detach().cpu().numpy().reshape(-1)  # [k_theta]
    theta = destandardize(theta_std, y_mean, y_scale)              # [k_theta] original scale

    return theta, theta_cols


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PI-GNN Demo (Model 1)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model once
        try:
            self.model, self.meta, self.scalers = load_model(OUTPUTS_DIR, self.device)
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            raise

        frm = ttk.Frame(root, padding=14)
        frm.grid(row=0, column=0, sticky="nsew")

        # SMILES
        ttk.Label(frm, text="SMILES").grid(row=0, column=0, sticky="w", pady=4)
        self.smiles_var = tk.StringVar(value="CCO")
        ttk.Entry(frm, textvariable=self.smiles_var, width=40).grid(row=0, column=1, sticky="ew", pady=4)

        # M
        ttk.Label(frm, text="M (Molar Mass)").grid(row=1, column=0, sticky="w", pady=4)
        self.m_var = tk.StringVar(value="46.07")
        ttk.Entry(frm, textvariable=self.m_var, width=20).grid(row=1, column=1, sticky="w", pady=4)

        # T
        ttk.Label(frm, text="T (Degrees Celsius)").grid(row=2, column=0, sticky="w", pady=4)
        self.t_var = tk.StringVar(value="298")
        ttk.Entry(frm, textvariable=self.t_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        # Predict button
        self.btn = ttk.Button(frm, text="Predict", command=self.on_predict)
        self.btn.grid(row=3, column=0, columnspan=2, pady=10)

        # Output
        ttk.Label(frm, text="Prediction").grid(row=4, column=0, sticky="nw", pady=4)
        self.out_text = tk.Text(frm, height=7, width=60)
        self.out_text.grid(row=4, column=1, sticky="nsew", pady=4)

        # layout weights
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(4, weight=1)

        # Footer note
        self.status = ttk.Label(frm, text=f"Device: {self.device} | Outputs: {OUTPUTS_DIR}")
        self.status.grid(row=5, column=0, columnspan=2, sticky="w", pady=4)

    def on_predict(self):
        smiles = self.smiles_var.get().strip()
        try:
            M = float(self.m_var.get().strip())
            T = float(self.t_var.get().strip())
        except ValueError:
            messagebox.showerror("Input error", "M and T must be numeric.")
            return

        self.out_text.delete("1.0", tk.END)

        try:
            theta, theta_cols = predict_single(
                self.model, self.meta, self.scalers, self.device,
                smiles=smiles, M=M, T=T
            )
        except Exception as e:
            messagebox.showerror("Prediction failed", str(e))
            return

        # display
        self.out_text.insert(tk.END, f"SMILES: {smiles}\n")
        self.out_text.insert(tk.END, f"M: {M}\nT: {T}\n\n")
        for i, name in enumerate(theta_cols):
            self.out_text.insert(tk.END, f"{name}: {float(theta[i]):.6f}\n")


def main():
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
