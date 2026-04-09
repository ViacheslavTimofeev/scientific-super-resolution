from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.eval.metrics import compute_metrics
from src.eval.tensors import align_image_channels
from src.runtime.device import resolve_device


Batch = Mapping[str, Tensor | str]


def _get_train_cfg(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    return config.get("train", {})


def _get_eval_cfg(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    return config.get("eval", {})


def _move_tensor_batch(
    batch: Batch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> tuple[Tensor, Tensor]:
    lr = batch["lr"]
    hr = batch["hr"]

    if not isinstance(lr, Tensor) or not isinstance(hr, Tensor):
        raise TypeError("Expected batch to contain tensor entries 'lr' and 'hr'.")

    return (
        lr.to(device=device, non_blocking=non_blocking),
        hr.to(device=device, non_blocking=non_blocking),
    )


def _build_autocast_context(*, device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()

    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _merge_metric_sums(
    running_sums: dict[str, float],
    batch_metrics: Mapping[str, float],
    *,
    batch_size: int,
) -> None:
    for name, value in batch_metrics.items():
        running_sums[name] = running_sums.get(name, 0.0) + value * batch_size


def _finalize_epoch_stats(
    *,
    total_loss: float,
    metric_sums: Mapping[str, float],
    num_samples: int,
) -> dict[str, float]:
    if num_samples == 0:
        raise ValueError("Dataloader is empty, cannot finalize epoch statistics.")

    stats = {"loss": total_loss / num_samples}
    for name, value in metric_sums.items():
        stats[name] = value / num_samples
    return stats


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    *,
    device: str | torch.device | None = None,
    config: Mapping[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    """
    Run a single training epoch.

    The function uses project YAML config values when available:
    - `device`
    - `train.amp`
    - `train.grad_clip_norm`
    - `eval.metric_names`
    """
    resolved_device = resolve_device(device, config)
    train_cfg = _get_train_cfg(config)
    eval_cfg = _get_eval_cfg(config)
    metric_names = tuple(eval_cfg.get("metric_names", ("psnr", "ssim")))
    use_amp = bool(train_cfg.get("amp", False)) and resolved_device.type == "cuda"
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    model.to(resolved_device)
    loss_fn.to(resolved_device)

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    num_samples = 0
    non_blocking = resolved_device.type == "cuda"

    for step, batch in enumerate(dataloader, start=1):
        lr, hr = _move_tensor_batch(
            batch,
            device=resolved_device,
            non_blocking=non_blocking,
        )
        batch_size = lr.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with _build_autocast_context(device=resolved_device, enabled=use_amp):
            prediction = model(lr)
            prediction, hr = align_image_channels(prediction, hr)
            loss = loss_fn(prediction, hr)

        scaler.scale(loss).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        detached_prediction = prediction.detach().float().clamp_(0.0, 1.0)
        detached_target = hr.detach().float()
        batch_metrics = compute_metrics(
            detached_prediction,
            detached_target,
            metric_names=metric_names,
        )

        total_loss += float(loss.detach().item()) * batch_size
        _merge_metric_sums(metric_sums, batch_metrics, batch_size=batch_size)
        num_samples += batch_size

    return _finalize_epoch_stats(
        total_loss=total_loss,
        metric_sums=metric_sums,
        num_samples=num_samples,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: str | torch.device | None = None,
    config: Mapping[str, Any] | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    """Run a single validation epoch using the same config conventions as training."""
    resolved_device = resolve_device(device, config)
    train_cfg = _get_train_cfg(config)
    eval_cfg = _get_eval_cfg(config)
    metric_names = tuple(eval_cfg.get("metric_names", ("psnr", "ssim")))
    use_amp = bool(train_cfg.get("amp", False)) and resolved_device.type == "cuda"
    model.eval()
    model.to(resolved_device)
    loss_fn.to(resolved_device)

    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    num_samples = 0
    non_blocking = resolved_device.type == "cuda"

    for step, batch in enumerate(dataloader, start=1):
        lr, hr = _move_tensor_batch(
            batch,
            device=resolved_device,
            non_blocking=non_blocking,
        )
        batch_size = lr.shape[0]

        with _build_autocast_context(device=resolved_device, enabled=use_amp):
            prediction = model(lr)
            prediction, hr = align_image_channels(prediction, hr)
            loss = loss_fn(prediction, hr)

        detached_prediction = prediction.detach().float().clamp_(0.0, 1.0)
        detached_target = hr.detach().float()
        batch_metrics = compute_metrics(
            detached_prediction,
            detached_target,
            metric_names=metric_names,
        )

        total_loss += float(loss.detach().item()) * batch_size
        _merge_metric_sums(metric_sums, batch_metrics, batch_size=batch_size)
        num_samples += batch_size

    return _finalize_epoch_stats(
        total_loss=total_loss,
        metric_sums=metric_sums,
        num_samples=num_samples,
    )
