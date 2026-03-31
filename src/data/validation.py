from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted(path for path in directory.iterdir() if is_image_file(path))
    if not files:
        raise FileNotFoundError(f"No image files found in: {directory}")
    return files


def validate_patch_size(*, patch_size: int | None, scale: int) -> None:
    if patch_size is not None and patch_size % scale != 0:
        raise ValueError(
            f"patch_size={patch_size} must be divisible by scale={scale}."
        )


def validate_crop_multiple(
    *,
    patch_size: int | None,
    scale: int,
    crop_multiple: int | None,
) -> None:
    if crop_multiple is None:
        return

    lr_patch = None if patch_size is None else patch_size // scale
    if lr_patch is not None and lr_patch % crop_multiple != 0:
        raise ValueError(
            "For transformer-style training, LR patch size must be divisible "
            f"by crop_multiple={crop_multiple}. "
            f"Got LR patch size {lr_patch}."
        )


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


def validate_paired_image_size(
    *,
    lr_image: Image.Image,
    hr_image: Image.Image,
    scale: int,
) -> None:
    lr_width, lr_height = lr_image.size
    expected_hr_size = (lr_width * scale, lr_height * scale)

    if hr_image.size != expected_hr_size:
        raise ValueError(
            f"Size mismatch for paired sample: LR={lr_image.size}, "
            f"HR={hr_image.size}, expected HR={expected_hr_size}."
        )
