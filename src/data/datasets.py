from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.data.transforms import (
    SampleTransform,
    build_paired_image_transform,
    build_synthetic_hr_transform,
)


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
        sample_transform: SampleTransform | None = None,
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

        self.image_transform = build_paired_image_transform(
            scale=self.scale,
            patch_size=self.patch_size,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
            random_rotate=self.random_rotate,
            crop_multiple=self.crop_multiple,
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
            return self.image_transform(lr_image, hr_image)

        if self.crop_multiple is not None:
            return self.image_transform(lr_image, hr_image)

        return self.image_transform(lr_image, hr_image)


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
        sample_transform: SampleTransform | None = None,
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
        self.hr_transform = build_synthetic_hr_transform(
            scale=self.scale,
            patch_size=self.patch_size,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
            random_rotate=self.random_rotate,
            crop_multiple=self.crop_multiple,
        )

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
        return self.hr_transform(hr_image)


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
