from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(theta_pred: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(theta_pred, theta_true)


def nonneg_penalty(theta_pred: torch.Tensor) -> torch.Tensor:
    # Penalize negative predictions softly
    return torch.mean(F.relu(-theta_pred))


def range_penalty(theta_pred: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # lo/hi are [k] tensors (standardized-space bounds if y is standardized)
    below = F.relu(lo - theta_pred)
    above = F.relu(theta_pred - hi)
    return torch.mean(below + above)
