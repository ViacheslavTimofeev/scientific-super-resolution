from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.data.validation import list_image_files


def match_lr_to_hr_name(lr_name: str, scale: int) -> str:
    suffix = f"x{scale}"
    if lr_name.endswith(suffix):
        return lr_name[: -len(suffix)]
    return lr_name


def build_hr_index(hr_paths: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in hr_paths:
        index[path.stem] = path
    return index


def build_paired_samples(
    *,
    lr_dir: Path,
    hr_dir: Path,
    scale: int,
) -> tuple[list[Path], list[tuple[Path, Path, str]]]:
    lr_paths = list_image_files(lr_dir)
    hr_index = build_hr_index(list_image_files(hr_dir))

    samples: list[tuple[Path, Path, str]] = []
    missing_hr: list[str] = []

    for lr_path in lr_paths:
        image_id = match_lr_to_hr_name(lr_path.stem, scale)
        hr_path = hr_index.get(image_id)
        if hr_path is None:
            missing_hr.append(lr_path.name)
            continue
        samples.append((lr_path, hr_path, image_id))

    if missing_hr:
        preview = ", ".join(missing_hr[:5])
        raise FileNotFoundError(
            "Could not find matching HR images for LR files: "
            f"{preview}"
        )

    if not samples:
        raise FileNotFoundError(
            f"No LR/HR pairs found in {lr_dir} and {hr_dir}."
        )

    return lr_paths, samples
