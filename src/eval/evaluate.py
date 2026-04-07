from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.data.dataloaders import build_eval_dataloader
from src.eval.metrics import align_image_channels, compute_metrics
from src.models.factory import build_model
from src.train.losses import build_loss
from src.train.loops import resolve_device


def _resolve_checkpoint_path(
    config: Mapping[str, Any],
    checkpoint_path: str | Path | None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    output_cfg = dict(config.get("output", {}))
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
    best_checkpoint_name = output_cfg.get("best_checkpoint_name", "best.pt")
    return checkpoint_dir / best_checkpoint_name


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a mapping. "
            f"Got {type(checkpoint)!r} from '{checkpoint_path}'."
        )

    return checkpoint


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    candidate_keys = ("model_state_dict", "params_ema", "params", "state_dict")

    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            state_dict = dict(value)
            break
    else:
        state_dict = dict(checkpoint)

    # Handle checkpoints saved from DataParallel/DistributedDataParallel wrappers.
    if state_dict and all(isinstance(name, str) for name in state_dict):
        if all(name.startswith("module.") for name in state_dict):
            state_dict = {
                name.removeprefix("module."): tensor
                for name, tensor in state_dict.items()
            }

    return state_dict


def _build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    if "kind" not in model_cfg:
        raise KeyError("Model config must contain 'kind'.")

    model_kind = str(model_cfg.pop("kind"))
    return build_model(model_kind, **model_cfg)


def load_model_and_checkpoint(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    resolved_device = resolve_device(device, config)
    resolved_checkpoint_path = _resolve_checkpoint_path(config, checkpoint_path)
    checkpoint = _load_checkpoint(resolved_checkpoint_path)

    model = _build_model_from_config(config)

    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise KeyError(
            "Checkpoint must contain 'model_state_dict' or be a raw state dict mapping."
        )

    model.load_state_dict(dict(state_dict))
    model.to(resolved_device)
    model.eval()

    checkpoint["checkpoint_path"] = str(resolved_checkpoint_path)
    return model, checkpoint, resolved_device


def _compute_per_image_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    metric_names: tuple[str, ...],
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for index in range(prediction.shape[0]):
        image_prediction = prediction[index : index + 1]
        image_target = target[index : index + 1]
        results.append(
            compute_metrics(
                image_prediction,
                image_target,
                metric_names=metric_names,
            )
        )
    return results


def _save_json(data: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


@torch.inference_mode()
def evaluate(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    return_per_image: bool = False,
    save_results_path: str | Path | None = None,
) -> dict[str, Any]:
    model, checkpoint, resolved_device = load_model_and_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    dataloader = build_eval_dataloader(config)

    train_cfg = dict(config.get("train", {}))
    eval_cfg = dict(config.get("eval", {}))
    metric_names = tuple(eval_cfg.get("metric_names", ("psnr", "ssim")))
    if not metric_names:
        raise ValueError("Evaluation metric list is empty.")

    loss_fn = build_loss(train_cfg.get("loss")).to(resolved_device)
    log_every = int(eval_cfg.get("log_every", train_cfg.get("log_every", 0)) or 0)

    print(
        f"Evaluating checkpoint: {checkpoint['checkpoint_path']} "
        f"on {len(dataloader)} batches"
    )

    total_loss = 0.0
    metric_sums = {name: 0.0 for name in metric_names}
    per_image_results: list[dict[str, Any]] = []
    num_images = 0
    non_blocking = resolved_device.type == "cuda"

    for step, batch in enumerate(dataloader, start=1):
        lr = batch["lr"]
        hr = batch["hr"]
        image_ids = batch["image_id"]
        lr_paths = batch["lr_path"]
        hr_paths = batch["hr_path"]

        if not isinstance(lr, Tensor) or not isinstance(hr, Tensor):
            raise TypeError("Expected batch to contain tensor entries 'lr' and 'hr'.")

        lr = lr.to(device=resolved_device, non_blocking=non_blocking)
        target = hr.to(device=resolved_device, non_blocking=non_blocking)

        prediction = model(lr)
        prediction, target = align_image_channels(prediction, target)
        loss = loss_fn(prediction, target)
        prediction = prediction.detach().float().clamp_(0.0, 1.0)
        target = target.detach().float()

        batch_metrics = _compute_per_image_metrics(
            prediction,
            target,
            metric_names=metric_names,
        )

        batch_size = prediction.shape[0]
        total_loss += float(loss.detach().item()) * batch_size

        batch_metric_means = {name: 0.0 for name in metric_names}
        for index, image_metrics in enumerate(batch_metrics):
            for name, value in image_metrics.items():
                metric_sums[name] += value
                batch_metric_means[name] += value

            num_images += 1
            if return_per_image or save_results_path is not None:
                per_image_results.append(
                    {
                        "image_id": str(image_ids[index]),
                        "lr_path": str(lr_paths[index]),
                        "hr_path": str(hr_paths[index]),
                        "metrics": image_metrics,
                    }
                )

        for name in batch_metric_means:
            batch_metric_means[name] /= batch_size

        if log_every > 0 and step % log_every == 0:
            metrics_text = ", ".join(
                f"{name}={batch_metric_means[name]:.4f}" for name in metric_names
            )
            print(
                f"eval step {step}/{len(dataloader)}: "
                f"loss={loss.detach().item():.4f}, {metrics_text}"
            )

    if num_images == 0:
        raise ValueError("Evaluation dataloader is empty.")

    average_loss = total_loss / num_images
    aggregated_metrics = {
        name: value / num_images for name, value in metric_sums.items()
    }
    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_images": num_images,
        "loss": average_loss,
        "metrics": aggregated_metrics,
    }

    summary = [f"Evaluation complete: num_images={num_images}", f"loss={average_loss:.4f}"]
    summary.extend(f"{name}={value:.4f}" for name, value in aggregated_metrics.items())
    print(" | ".join(summary))

    if return_per_image:
        results["per_image"] = per_image_results

    if save_results_path is not None:
        serializable_results = dict(results)
        serializable_results["per_image"] = per_image_results
        _save_json(serializable_results, save_results_path)
        results["results_path"] = str(Path(save_results_path))

    return results
