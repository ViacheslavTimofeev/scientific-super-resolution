from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence
import random

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted(path for path in directory.iterdir() if _is_image_file(path))
    if not files:
        raise FileNotFoundError(f"No image files found in: {directory}")
    return files


def _to_tensor(image: Image.Image) -> Tensor:
    return to_tensor(image)


def _center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    if crop_size > width or crop_size > height:
        raise ValueError(
            f"Crop size {crop_size} is larger than image size {(width, height)}."
        )

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return (left, top, left + crop_size, top + crop_size)


def _aligned_crop_size(
    width: int,
    height: int,
    scale: int,
    extra_multiple: int | None = None,
) -> int:
    crop_size = min(width, height)
    divisor = scale * (extra_multiple or 1)
    crop_size = (crop_size // divisor) * divisor
    if crop_size <= 0:
        raise ValueError(
            f"Image size {(width, height)} is too small for scale={scale} and "
            f"extra_multiple={extra_multiple}."
        )
    return crop_size


def _match_lr_to_hr_name(lr_name: str, scale: int) -> str:
    suffix = f"x{scale}"
    if lr_name.endswith(suffix):
        return lr_name[: -len(suffix)]
    return lr_name


def _build_hr_index(hr_paths: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in hr_paths:
        index[path.stem] = path
    return index


@dataclass(frozen=True)
class SuperResolutionSample:
    lr: Tensor
    hr: Tensor
    image_id: str
    lr_path: str
    hr_path: str


class PairedSuperResolutionDataset(Dataset[dict[str, Tensor | str]]):
    """
    Generic SR dataset for paired LR/HR image folders.

    CNN and transformer models can share the same dataset because the core
    supervision target is identical. The main difference is usually the crop
    alignment: transformers often need LR patches divisible by window_size.
    """

    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        *,
        scale: int,
        split: Literal["train", "valid", "test"] = "train",
        patch_size: int | None = None,
        image_mode: Literal["L", "RGB"] = "L",
        random_crop: bool | None = None,
        random_flip: bool = False,
        random_rotate: bool = False,
        crop_multiple: int | None = None,
        sample_transform: Callable[[dict[str, Tensor | str]], dict[str, Tensor | str]] | None = None,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.split = split
        self.patch_size = patch_size
        self.image_mode = image_mode
        self.random_crop = split == "train" if random_crop is None else random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.crop_multiple = crop_multiple
        self.sample_transform = sample_transform

        if self.patch_size is not None and self.patch_size % self.scale != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by scale={scale}."
            )

        if self.crop_multiple is not None:
            lr_patch = None if self.patch_size is None else self.patch_size // self.scale
            if lr_patch is not None and lr_patch % self.crop_multiple != 0:
                raise ValueError(
                    "For transformer-style training, LR patch size must be divisible "
                    f"by crop_multiple={self.crop_multiple}. "
                    f"Got LR patch size {lr_patch}."
                )

        self.lr_paths = _list_image_files(self.lr_dir)
        hr_index = _build_hr_index(_list_image_files(self.hr_dir))

        self.samples: list[tuple[Path, Path, str]] = []
        missing_hr: list[str] = []

        for lr_path in self.lr_paths:
            image_id = _match_lr_to_hr_name(lr_path.stem, self.scale)
            hr_path = hr_index.get(image_id)
            if hr_path is None:
                missing_hr.append(lr_path.name)
                continue
            self.samples.append((lr_path, hr_path, image_id))

        if missing_hr:
            preview = ", ".join(missing_hr[:5])
            raise FileNotFoundError(
                "Could not find matching HR images for LR files: "
                f"{preview}"
            )

        if not self.samples:
            raise FileNotFoundError(
                f"No LR/HR pairs found in {self.lr_dir} and {self.hr_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        lr_path, hr_path, image_id = self.samples[index]

        lr_image = Image.open(lr_path).convert(self.image_mode)
        hr_image = Image.open(hr_path).convert(self.image_mode)

        lr_image, hr_image = self._prepare_pair(lr_image, hr_image)
        lr_tensor = _to_tensor(lr_image)
        hr_tensor = _to_tensor(hr_image)

        sample: dict[str, Tensor | str] = {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "image_id": image_id,
            "lr_path": str(lr_path),
            "hr_path": str(hr_path),
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def _prepare_pair(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        lr_width, lr_height = lr_image.size
        hr_width, hr_height = hr_image.size

        expected_hr_size = (lr_width * self.scale, lr_height * self.scale)
        if (hr_width, hr_height) != expected_hr_size:
            raise ValueError(
                f"Size mismatch for paired sample: LR={lr_image.size}, "
                f"HR={hr_image.size}, expected HR={expected_hr_size}."
            )

        if self.patch_size is not None:
            return self._crop_patch_pair(lr_image, hr_image)

        if self.crop_multiple is not None:
            return self._crop_full_pair_to_alignment(lr_image, hr_image)

        return lr_image, hr_image

    def _crop_patch_pair(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        hr_patch = self.patch_size
        lr_patch = hr_patch // self.scale
        lr_width, lr_height = lr_image.size

        if lr_patch > lr_width or lr_patch > lr_height:
            raise ValueError(
                f"LR patch size {lr_patch} is larger than LR image size {lr_image.size}."
            )

        if self.random_crop:
            left = random.randint(0, lr_width - lr_patch)
            top = random.randint(0, lr_height - lr_patch)
        else:
            left, top, _, _ = _center_crop_box(lr_width, lr_height, lr_patch)

        lr_box = (left, top, left + lr_patch, top + lr_patch)
        hr_box = (
            left * self.scale,
            top * self.scale,
            (left + lr_patch) * self.scale,
            (top + lr_patch) * self.scale,
        )

        lr_crop = lr_image.crop(lr_box)
        hr_crop = hr_image.crop(hr_box)
        return self._augment_pair(lr_crop, hr_crop)

    def _crop_full_pair_to_alignment(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        lr_width, lr_height = lr_image.size
        aligned_lr_size = _aligned_crop_size(
            width=lr_width,
            height=lr_height,
            scale=1,
            extra_multiple=self.crop_multiple,
        )
        left, top, right, bottom = _center_crop_box(
            lr_width,
            lr_height,
            aligned_lr_size,
        )

        lr_crop = lr_image.crop((left, top, right, bottom))
        hr_crop = hr_image.crop(
            (
                left * self.scale,
                top * self.scale,
                right * self.scale,
                bottom * self.scale,
            )
        )
        return lr_crop, hr_crop

    def _augment_pair(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if self.random_flip and random.random() < 0.5:
            lr_image = lr_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hr_image = hr_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.random_flip and random.random() < 0.5:
            lr_image = lr_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            hr_image = hr_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if self.random_rotate:
            rotation = random.choice((0, 90, 180, 270))
            if rotation:
                lr_image = lr_image.rotate(rotation)
                hr_image = hr_image.rotate(rotation)

        return lr_image, hr_image


class SyntheticSuperResolutionDataset(Dataset[dict[str, Tensor | str]]):
    """
    SR dataset that starts from HR images and synthesizes LR on the fly.

    Useful when the project later wants a unified pipeline for arbitrary
    downsampling policies instead of only pre-generated LR folders.
    """

    def __init__(
        self,
        hr_dir: str | Path,
        *,
        scale: int,
        split: Literal["train", "valid", "test"] = "train",
        patch_size: int | None = None,
        image_mode: Literal["L", "RGB"] = "L",
        random_crop: bool | None = None,
        random_flip: bool = False,
        random_rotate: bool = False,
        crop_multiple: int | None = None,
        downsample_resample: int = Image.Resampling.BICUBIC,
        sample_transform: Callable[[dict[str, Tensor | str]], dict[str, Tensor | str]] | None = None,
    ) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.split = split
        self.patch_size = patch_size
        self.image_mode = image_mode
        self.random_crop = split == "train" if random_crop is None else random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.crop_multiple = crop_multiple
        self.downsample_resample = downsample_resample
        self.sample_transform = sample_transform

        if self.patch_size is not None and self.patch_size % self.scale != 0:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by scale={scale}."
            )

        if self.crop_multiple is not None:
            lr_patch = None if self.patch_size is None else self.patch_size // self.scale
            if lr_patch is not None and lr_patch % self.crop_multiple != 0:
                raise ValueError(
                    "For transformer-style training, LR patch size must be divisible "
                    f"by crop_multiple={self.crop_multiple}. "
                    f"Got LR patch size {lr_patch}."
                )

        self.hr_paths = _list_image_files(self.hr_dir)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        hr_path = self.hr_paths[index]
        hr_image = Image.open(hr_path).convert(self.image_mode)
        hr_image = self._prepare_hr_image(hr_image)

        lr_size = (hr_image.width // self.scale, hr_image.height // self.scale)
        lr_image = hr_image.resize(lr_size, resample=self.downsample_resample)

        lr_tensor = _to_tensor(lr_image)
        hr_tensor = _to_tensor(hr_image)

        sample: dict[str, Tensor | str] = {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "image_id": hr_path.stem,
            "lr_path": "",
            "hr_path": str(hr_path),
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def _prepare_hr_image(self, hr_image: Image.Image) -> Image.Image:
        width, height = hr_image.size

        if self.patch_size is not None:
            crop_size = self.patch_size
        else:
            crop_size = _aligned_crop_size(
                width=width,
                height=height,
                scale=self.scale,
                extra_multiple=self.crop_multiple,
            )

        if self.random_crop and crop_size < min(width, height):
            max_left = width - crop_size
            max_top = height - crop_size
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
            box = (left, top, left + crop_size, top + crop_size)
        else:
            box = _center_crop_box(width, height, crop_size)

        hr_crop = hr_image.crop(box)

        if self.random_flip and random.random() < 0.5:
            hr_crop = hr_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.random_flip and random.random() < 0.5:
            hr_crop = hr_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.random_rotate:
            rotation = random.choice((0, 90, 180, 270))
            if rotation:
                hr_crop = hr_crop.rotate(rotation)

        return hr_crop


def build_paired_sr_dataset(
    *,
    lr_dir: str | Path,
    hr_dir: str | Path,
    scale: int,
    split: Literal["train", "valid", "test"] = "train",
    patch_size: int | None = None,
    image_mode: Literal["L", "RGB"] = "L",
    model_kind: Literal["cnn", "transformer"] = "cnn",
    random_flip: bool = False,
    random_rotate: bool = False,
    window_size: int | None = None,
) -> PairedSuperResolutionDataset:
    crop_multiple = window_size if model_kind == "transformer" else None
    return PairedSuperResolutionDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale=scale,
        split=split,
        patch_size=patch_size,
        image_mode=image_mode,
        random_flip=random_flip,
        random_rotate=random_rotate,
        crop_multiple=crop_multiple,
    )
