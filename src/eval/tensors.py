from __future__ import annotations

from torch import Tensor


def align_image_channels(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.ndim != 4 or target.ndim != 4:
        return prediction, target

    prediction_channels = prediction.shape[1]
    target_channels = target.shape[1]
    if prediction_channels == target_channels:
        return prediction, target

    # Pretrained RGB SR models may output 3 channels even when the dataset
    # target is grayscale. In that case we compare in luminance space.
    if prediction_channels == 3 and target_channels == 1:
        return prediction.mean(dim=1, keepdim=True), target
    if prediction_channels == 1 and target_channels == 3:
        return prediction, target.mean(dim=1, keepdim=True)

    return prediction, target
