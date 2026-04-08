from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def resolve_device(
    device: str | torch.device | None,
    config: Mapping[str, Any] | None,
) -> torch.device:
    requested_device = device
    if requested_device is None and config is not None and config.get("device") is not None:
        requested_device = str(config["device"])

    if requested_device is None:
        raise RuntimeError(
            "Device is not configured. Set `device` explicitly in the call "
            "or provide `device` in the config."
        )

    resolved_device = torch.device(requested_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{requested_device}' was requested, but CUDA is not available."
        )

    return resolved_device
