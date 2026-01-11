from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import ChemThetaDataset, load_tables, read_split_ids
from losses import mse_loss, nonneg_penalty
from model import PIGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: PIGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = mse_loss(out.theta_pred, batch.y)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 1e9


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs_model1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tables + schema
    master_df, targets_df, schema = load_tables(
        data_dir=cfg.data_dir,
        master_csv=cfg.master_csv,
        targets_csv=cfg.targets_csv,
        id_col=cfg.id_col,
        smiles_col=cfg.smiles_col,
        theta_prefix=cfg.theta_prefix,
        scalar_cols=cfg.scalar_cols,
    )

    # Load splits
    splits_path = cfg.data_dir / cfg.splits_dir
    train_ids = read_split_ids(splits_path / "train.txt")
    val_ids = read_split_ids(splits_path / "val.txt")
    test_ids = read_split_ids(splits_path / "test.txt")

    # Build datasets (fit scalers on train only)
    train_ds = ChemThetaDataset(master_df, targets_df, schema, train_ids)
    val_ds = ChemThetaDataset(master_df, targets_df, schema, val_ids,
                              scaler_u=train_ds.scaler_u, scaler_y=train_ds.scaler_y)
    test_ds = ChemThetaDataset(master_df, targets_df, schema, test_ids,
                               scaler_u=train_ds.scaler_u, scaler_y=train_ds.scaler_y)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Infer dims from one sample
    sample = train_ds.get(0)
    node_dim = sample.x.size(-1)
    phys_dim = sample.u.numel()
    k_theta = sample.y.numel()

    model = PIGNN(
        node_dim=node_dim,
        phys_dim=phys_dim,
        gnn_hidden=cfg.gnn_hidden,
        gnn_layers=cfg.gnn_layers,
        mlp_hidden=cfg.mlp_hidden,
        z_dim=cfg.z_dim,
        k_theta=k_theta,
        dropout=cfg.dropout,
        pool=cfg.pool,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best_model.pt"
    patience_left = cfg.patience

    # Save schema/scalers info for inference
    meta = {
        "id_col": cfg.id_col,
        "smiles_col": cfg.smiles_col,
        "scalar_cols": schema.scalar_cols,
        "theta_cols": schema.theta_cols,
        "node_dim": int(node_dim),
        "phys_dim": int(phys_dim),
        "k_theta": int(k_theta),
        "z_dim": int(cfg.z_dim),
        "pool": cfg.pool,
        "gnn_hidden": cfg.gnn_hidden,
        "gnn_layers": cfg.gnn_layers,
        "mlp_hidden": cfg.mlp_hidden,
        "dropout": cfg.dropout,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save scalers (simple JSON for mean/std)
    scaler_payload = {
        "u_mean": train_ds.scaler_u.mean_.tolist(),
        "u_scale": train_ds.scaler_u.scale_.tolist(),
        "y_mean": train_ds.scaler_y.mean_.tolist(),
        "y_scale": train_ds.scaler_y.scale_.tolist(),
    }
    (out_dir / "scalers.json").write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", leave=False)
        train_losses = []

        for batch in pbar:
            batch = batch.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(batch)

            loss = mse_loss(out.theta_pred, batch.y)

            # Optional physics penalty: non-negativity in standardized space can still help
            if cfg.lambda_nonneg > 0:
                loss = loss + cfg.lambda_nonneg * nonneg_penalty(out.theta_pred)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": np.mean(train_losses)})

        val_loss = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] train={np.mean(train_losses):.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_left = cfg.patience
            torch.save({"state_dict": model.state_dict()}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"Best val={best_val:.5f} | test={test_loss:.5f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
