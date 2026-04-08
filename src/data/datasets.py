from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from src.data.pairs import build_paired_samples
from src.data.transforms import (
    SampleTransform,
    build_paired_image_transform,
)
from src.data.validation import (
    validate_crop_multiple,
    validate_paired_image_size,
    validate_patch_size,
)


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

        validate_patch_size(patch_size=self.patch_size, scale=self.scale)
        validate_crop_multiple(
            patch_size=self.patch_size,
            scale=self.scale,
            crop_multiple=self.crop_multiple,
        )

        self.lr_paths, self.samples = build_paired_samples(
            lr_dir=self.lr_dir,
            hr_dir=self.hr_dir,
            scale=self.scale,
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
        lr_tensor = tv_to_tensor(lr_image)
        hr_tensor = tv_to_tensor(hr_image)

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
        validate_paired_image_size(
            lr_image=lr_image,
            hr_image=hr_image,
            scale=self.scale,
        )

        if self.patch_size is not None:
            return self.image_transform(lr_image, hr_image)

        if self.crop_multiple is not None:
            return self.image_transform(lr_image, hr_image)

        return self.image_transform(lr_image, hr_image)


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
