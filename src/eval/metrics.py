from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from src.data.validation import validate_image_tensors


MetricFn = Callable[[Tensor, Tensor], float]


def psnr(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
) -> float:
    prediction, target = validate_image_tensors(prediction, target)
    score = peak_signal_noise_ratio(prediction, target, data_range=data_range)
    return float(score.item())


def ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")

    prediction, target = validate_image_tensors(prediction, target)

    if min(prediction.shape[-2:]) < kernel_size:
        raise ValueError(
            "Image spatial size must be at least as large as kernel_size. "
            f"Got image size {tuple(prediction.shape[-2:])} and kernel_size={kernel_size}."
        )

    score = structural_similarity_index_measure(
        prediction,
        target,
        data_range=data_range,
        kernel_size=kernel_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
    )
    return float(score.item())


METRICS: dict[str, MetricFn] = {
    "psnr": psnr,
    "ssim": ssim,
}


def compute_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    metric_names: list[str] | tuple[str, ...] = ("psnr", "ssim"),
) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in metric_names:
        metric = METRICS.get(name)
        if metric is None:
            available = ", ".join(sorted(METRICS))
            raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}.")
        results[name] = metric(prediction, target)
    return results
