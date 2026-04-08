from __future__ import annotations
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
)
from torch.utils.tensorboard import SummaryWriter

from src.data.dataloaders import build_train_dataloaders
from src.models.factory import build_model
from src.runtime.device import resolve_device
from src.train.loops import train_one_epoch, validate_one_epoch
from src.train.losses import build_loss


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping config in '{config_path}', got {type(config)!r}.")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    optimizer_cfg: Mapping[str, Any] | None,
) -> Optimizer:
    optimizer_cfg = dict(optimizer_cfg or {})
    optimizer_name = str(optimizer_cfg.get("name", "adam")).lower()
    optimizer_kwargs = {
        key: value for key, value in optimizer_cfg.items() if key != "name"
    }

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), **optimizer_kwargs)

    raise ValueError(
        f"Unknown optimizer '{optimizer_name}'. Available optimizers: adamw."
    )


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Mapping[str, Any] | None,
) -> LRScheduler | ReduceLROnPlateau | None:
    if not scheduler_cfg:
        return None

    scheduler_cfg = dict(scheduler_cfg)
    scheduler_name = str(scheduler_cfg.get("name", "")).lower()
    scheduler_kwargs = {
        key: value for key, value in scheduler_cfg.items() if key != "name"
    }

    if scheduler_name == "cosine_annealing":
        return CosineAnnealingLR(optimizer, **scheduler_kwargs)
    if scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, **scheduler_kwargs, factor=0.5)
    if scheduler_name == "none":
        return None

    raise ValueError(
        "Unknown scheduler "
        f"'{scheduler_name}'. Available schedulers: cosine_annealing, reduce_on_plateau, none."
    )


def _is_higher_better(metric_name: str) -> bool:
    return metric_name.lower() not in {"loss", "l1", "mse", "mae"}


def _is_better_metric(
    current_value: float,
    best_value: float | None,
    *,
    metric_name: str,
) -> bool:
    if best_value is None:
        return True

    if _is_higher_better(metric_name):
        return current_value > best_value

    return current_value < best_value


def _prepare_output_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output_cfg = dict(config.get("output", {}))
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / output_cfg.get("best_checkpoint_name", "best.pt")
    last_checkpoint_path = checkpoint_dir / output_cfg.get("last_checkpoint_name", "last.pt")

    return checkpoint_dir, best_checkpoint_path, last_checkpoint_path


def _make_checkpoint(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler | None,
    config: Mapping[str, Any],
    epoch: int,
    train_stats: Mapping[str, float],
    valid_stats: Mapping[str, float] | None,
    best_metric_name: str,
    best_metric_value: float | None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_stats": dict(train_stats),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if valid_stats is not None:
        checkpoint["valid_stats"] = dict(valid_stats)

    return checkpoint


def _save_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), path)


def _prepare_tensorboard_writer(
    config: Mapping[str, Any],
) -> tuple[SummaryWriter, Path]:
    output_cfg = dict(config.get("output", {}))
    output_dir = Path(output_cfg.get("dir", "./outputs"))
    experiment_name = str(config.get("experiment_name", "experiment"))
    log_dir = output_dir / "tensorboard" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    config_yaml = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    writer.add_text("config/yaml", f"```yaml\n{config_yaml}\n```")

    return writer, log_dir


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved_device = resolve_device(device=None, config=config)
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    loss_cfg = train_cfg.get("loss")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_loader, valid_loader = build_train_dataloaders(config)

    model_kind = model_cfg.pop("kind")
    model = build_model(model_kind, **model_cfg).to(resolved_device)
    loss_fn = build_loss(loss_cfg).to(resolved_device)
    optimizer = build_optimizer(model, train_cfg.get("optimizer"))
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler"))

    use_amp = bool(train_cfg.get("amp", False)) and resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(train_cfg.get("epochs", 1))
    validate_every = int(train_cfg.get("validate_every", 1))
    save_best_metric = str(train_cfg.get("save_best_metric", "psnr"))

    checkpoint_dir, best_checkpoint_path, last_checkpoint_path = _prepare_output_paths(config)
    writer, tensorboard_log_dir = _prepare_tensorboard_writer(config)
    print(f"Saving checkpoints to: {checkpoint_dir}")
    print(f"Writing TensorBoard logs to: {tensorboard_log_dir}")

    best_metric_value: float | None = None
    history: list[dict[str, Any]] = []

    try:
        for epoch in range(1, epochs + 1):
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device=resolved_device,
                config=config,
                scaler=scaler,
                epoch=epoch,
            )

            valid_stats: dict[str, float] | None = None
            if validate_every > 0 and epoch % validate_every == 0:
                valid_stats = validate_one_epoch(
                    model,
                    valid_loader,
                    loss_fn,
                    device=resolved_device,
                    config=config,
                    epoch=epoch,
                )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    metric_source = valid_stats if valid_stats is not None else train_stats
                    if save_best_metric not in metric_source:
                        available = ", ".join(sorted(metric_source))
                        raise KeyError(
                            f"Metric '{save_best_metric}' is not available for scheduler step. "
                            f"Available metrics: {available}."
                        )
                    scheduler.step(metric_source[save_best_metric])
                else:
                    scheduler.step()

            metric_source = valid_stats if valid_stats is not None else train_stats
            if save_best_metric not in metric_source:
                available = ", ".join(sorted(metric_source))
                raise KeyError(
                    f"Metric '{save_best_metric}' is not available for checkpoint selection. "
                    f"Available metrics: {available}."
                )

            current_metric_value = metric_source[save_best_metric]
            is_best = _is_better_metric(
                current_metric_value,
                best_metric_value,
                metric_name=save_best_metric,
            )
            if is_best:
                best_metric_value = current_metric_value

            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train": dict(train_stats),
                "valid": dict(valid_stats) if valid_stats is not None else None,
            }
            history.append(epoch_record)

            writer.add_scalar("train/lr", current_lr, epoch)
            for name, value in train_stats.items():
                writer.add_scalar(f"train/{name}", value, epoch)
            if valid_stats is not None:
                for name, value in valid_stats.items():
                    writer.add_scalar(f"valid/{name}", value, epoch)
            writer.add_scalar(f"best/{save_best_metric}", float(best_metric_value), epoch)
            writer.flush()

            summary = [f"Epoch {epoch}/{epochs}", f"lr={current_lr:.6g}"]
            summary.extend(f"train_{name}={value:.4f}" for name, value in train_stats.items())
            if valid_stats is not None:
                summary.extend(f"valid_{name}={value:.4f}" for name, value in valid_stats.items())
            summary.append(f"{save_best_metric}={current_metric_value:.4f}")
            print(" | ".join(summary))

            checkpoint = _make_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                train_stats=train_stats,
                valid_stats=valid_stats,
                best_metric_name=save_best_metric,
                best_metric_value=best_metric_value,
            )
            _save_checkpoint(checkpoint, last_checkpoint_path)

            if is_best:
                _save_checkpoint(checkpoint, best_checkpoint_path)
                print(
                    f"Saved best checkpoint at epoch {epoch}: "
                    f"{save_best_metric}={best_metric_value:.4f}"
                )
    finally:
        writer.close()

    return {
        "best_metric_name": save_best_metric,
        "best_metric_value": best_metric_value,
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
    }
