from __future__ import annotations

from pathlib import Path

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
