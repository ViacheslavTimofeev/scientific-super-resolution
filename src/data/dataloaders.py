from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader

from src.data.datasets import (
    PairedSuperResolutionDataset,
    build_paired_sr_dataset,
)


def _build_dataloader(
    dataset: PairedSuperResolutionDataset,
    loader_cfg: Mapping[str, Any] | None = None,
) -> DataLoader:
    loader_cfg = dict(loader_cfg or {})
    return DataLoader(
        dataset,
        batch_size=loader_cfg.get("batch_size", 1),
        shuffle=loader_cfg.get("shuffle", False),
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", False),
        drop_last=loader_cfg.get("drop_last", False),
    )


def build_paired_dataloader(
    *,
    lr_dir: str,
    hr_dir: str,
    scale: int,
    split: str = "train",
    patch_size: int | None = None,
    image_mode: str = "L",
    model_kind: str = "cnn",
    random_flip: bool = False,
    random_rotate: bool = False,
    window_size: int | None = None,
    loader_cfg: Mapping[str, Any] | None = None,
) -> DataLoader:
    dataset = build_paired_sr_dataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale=scale,
        split=split,
        patch_size=patch_size,
        image_mode=image_mode,
        model_kind=model_kind,
        random_flip=random_flip,
        random_rotate=random_rotate,
        window_size=window_size,
    )
    return _build_dataloader(dataset, loader_cfg)


def build_train_dataloaders(config: Mapping[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))

    shared_dataset_cfg = {
        "scale": data_cfg["scale"],
        "image_mode": data_cfg.get("image_mode", "L"),
        "model_kind": model_cfg.get("kind", "cnn"),
        "window_size": model_cfg.get("window_size"),
    }

    train_loader = build_paired_dataloader(
        lr_dir=data_cfg["train"]["lr_dir"],
        hr_dir=data_cfg["train"]["hr_dir"],
        split="train",
        patch_size=data_cfg.get("patch_size"),
        random_flip=data_cfg.get("random_flip", False),
        random_rotate=data_cfg.get("random_rotate", False),
        loader_cfg=data_cfg["train"],
        **shared_dataset_cfg,
    )
    valid_loader = build_paired_dataloader(
        lr_dir=data_cfg["valid"]["lr_dir"],
        hr_dir=data_cfg["valid"]["hr_dir"],
        split="valid",
        patch_size=None,
        random_flip=False,
        random_rotate=False,
        loader_cfg=data_cfg["valid"],
        **shared_dataset_cfg,
    )
    return train_loader, valid_loader


def build_eval_dataloader(config: Mapping[str, Any]) -> DataLoader:
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = dict(data_cfg.get("dataset") or data_cfg.get("valid") or {})

    return build_paired_dataloader(
        lr_dir=dataset_cfg["lr_dir"],
        hr_dir=dataset_cfg["hr_dir"],
        scale=data_cfg["scale"],
        split="valid",
        patch_size=None,
        image_mode=data_cfg.get("image_mode", "L"),
        model_kind=model_cfg.get("kind", "cnn"),
        window_size=model_cfg.get("window_size"),
        random_flip=False,
        random_rotate=False,
        loader_cfg=dataset_cfg,
    )
