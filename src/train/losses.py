from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
}


def build_loss(loss_cfg: str | Mapping[str, Any] | None = None) -> nn.Module:
    """Build a loss module from a string name or config mapping."""
    if loss_cfg is None:
        name = "l1"
        params: dict[str, Any] = {}
    elif isinstance(loss_cfg, str):
        name = loss_cfg
        params = {}
    else:
        name = str(loss_cfg.get("name", "l1"))
        params = {key: value for key, value in loss_cfg.items() if key != "name"}

    loss_cls = _LOSS_REGISTRY.get(name.lower())
    if loss_cls is None:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available}")

    return loss_cls(**params)
