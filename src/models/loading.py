from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.factory import build_model
from src.runtime.device import resolve_device


def resolve_checkpoint_path(
    config: Mapping[str, Any],
    checkpoint_path: str | Path | None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    output_cfg = dict(config.get("output", {}))
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
    best_checkpoint_name = output_cfg.get("best_checkpoint_name", "best.pt")
    return checkpoint_dir / best_checkpoint_name


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved_checkpoint_path}")

    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a mapping. "
            f"Got {type(checkpoint)!r} from '{resolved_checkpoint_path}'."
        )

    checkpoint["checkpoint_path"] = str(resolved_checkpoint_path)
    return checkpoint


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    candidate_keys = ("model_state_dict", "params_ema", "params", "state_dict")

    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            state_dict = dict(value)
            break
    else:
        state_dict = dict(checkpoint)

    if state_dict and all(isinstance(name, str) for name in state_dict):
        if all(name.startswith("module.") for name in state_dict):
            state_dict = {
                name.removeprefix("module."): tensor
                for name, tensor in state_dict.items()
            }

    return state_dict


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    if "kind" not in model_cfg:
        raise KeyError("Model config must contain 'kind'.")

    model_kind = str(model_cfg.pop("kind"))
    return build_model(model_kind, **model_cfg)


def prepare_model_for_inference(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    device: str | torch.device | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    resolved_device = resolve_device(device, config)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(resolved_device)
    return model, checkpoint, resolved_device


def load_model_and_checkpoint(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    resolved_checkpoint_path = resolve_checkpoint_path(config, checkpoint_path)
    model = build_model_from_config(config)
    return prepare_model_for_inference(
        model,
        checkpoint_path=resolved_checkpoint_path,
        device=device,
        config=config,
    )
