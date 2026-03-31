from __future__ import annotations

from collections.abc import Callable
import random

from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor as tv_to_tensor


Sample = dict[str, Tensor | str]
SampleTransform = Callable[[Sample], Sample]
PairedImageTransform = Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]
HRImageTransform = Callable[[Image.Image], Image.Image]


def compose_sample_transforms(*transforms: SampleTransform | None) -> SampleTransform | None:
    active_transforms = [transform for transform in transforms if transform is not None]
    if not active_transforms:
        return None

    def _composed(sample: Sample) -> Sample:
        for transform in active_transforms:
            sample = transform(sample)
        return sample

    return _composed


def to_tensor(image: Image.Image) -> Tensor:
    return tv_to_tensor(image)


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


def _apply_shared_augmentations(
    first: Image.Image,
    second: Image.Image,
    *,
    random_flip: bool,
    random_rotate: bool,
) -> tuple[Image.Image, Image.Image]:
    if random_flip and random.random() < 0.5:
        first = first.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        second = second.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random_flip and random.random() < 0.5:
        first = first.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        second = second.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if random_rotate:
        rotation = random.choice((0, 90, 180, 270))
        if rotation:
            first = first.rotate(rotation)
            second = second.rotate(rotation)

    return first, second


def build_paired_image_transform(
    *,
    scale: int,
    patch_size: int | None,
    random_crop: bool,
    random_flip: bool,
    random_rotate: bool,
    crop_multiple: int | None,
) -> PairedImageTransform:
    def _transform(
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        lr_width, lr_height = lr_image.size

        if patch_size is not None:
            hr_patch = patch_size
            lr_patch = hr_patch // scale

            if lr_patch > lr_width or lr_patch > lr_height:
                raise ValueError(
                    f"LR patch size {lr_patch} is larger than LR image size {lr_image.size}."
                )

            if random_crop:
                left = random.randint(0, lr_width - lr_patch)
                top = random.randint(0, lr_height - lr_patch)
            else:
                left, top, _, _ = _center_crop_box(lr_width, lr_height, lr_patch)

            lr_image = lr_image.crop((left, top, left + lr_patch, top + lr_patch))
            hr_image = hr_image.crop(
                (
                    left * scale,
                    top * scale,
                    (left + lr_patch) * scale,
                    (top + lr_patch) * scale,
                )
            )
        elif crop_multiple is not None:
            aligned_lr_size = _aligned_crop_size(
                width=lr_width,
                height=lr_height,
                scale=1,
                extra_multiple=crop_multiple,
            )
            left, top, right, bottom = _center_crop_box(
                lr_width,
                lr_height,
                aligned_lr_size,
            )
            lr_image = lr_image.crop((left, top, right, bottom))
            hr_image = hr_image.crop(
                (
                    left * scale,
                    top * scale,
                    right * scale,
                    bottom * scale,
                )
            )

        return _apply_shared_augmentations(
            lr_image,
            hr_image,
            random_flip=random_flip,
            random_rotate=random_rotate,
        )

    return _transform


def build_synthetic_hr_transform(
    *,
    scale: int,
    patch_size: int | None,
    random_crop: bool,
    random_flip: bool,
    random_rotate: bool,
    crop_multiple: int | None,
) -> HRImageTransform:
    def _transform(hr_image: Image.Image) -> Image.Image:
        width, height = hr_image.size

        if patch_size is not None:
            crop_size = patch_size
        else:
            crop_size = _aligned_crop_size(
                width=width,
                height=height,
                scale=scale,
                extra_multiple=crop_multiple,
            )

        if random_crop and crop_size < min(width, height):
            max_left = width - crop_size
            max_top = height - crop_size
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
            box = (left, top, left + crop_size, top + crop_size)
        else:
            box = _center_crop_box(width, height, crop_size)

        hr_image = hr_image.crop(box)
        hr_image, _ = _apply_shared_augmentations(
            hr_image,
            hr_image.copy(),
            random_flip=random_flip,
            random_rotate=random_rotate,
        )
        return hr_image

    return _transform
