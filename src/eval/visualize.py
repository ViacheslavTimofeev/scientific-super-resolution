from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import Tensor, nn

from src.data.dataloaders import build_eval_dataloader
from src.eval.metrics import align_image_channels, compute_metrics
from src.models.factory import build_model
from src.models.loading import load_model_and_checkpoint


def _tensor_to_pil_image(image: Tensor) -> Image.Image:
    image = image.detach().float().clamp(0.0, 1.0).cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}.")

    if image.shape[0] == 1:
        array = (image.squeeze(0) * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(array, mode="L").convert("RGB")

    if image.shape[0] == 3:
        array = (image.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(array, mode="RGB")

    raise ValueError(
        "Only 1-channel and 3-channel images are supported for visualization. "
        f"Got {image.shape[0]} channels."
    )


def _fit_image_to_cell(image: Image.Image, *, cell_size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(image, cell_size, color=(255, 255, 255), centering=(0.5, 0.5))


def _format_metric_lines(
    title: str,
    metrics: Mapping[str, float] | None = None,
    *,
    image_id: str | None = None,
) -> str:
    lines = [title]
    if image_id:
        lines.append(image_id)
    if metrics is not None:
        lines.extend(f"{name.upper()}: {value:.3f}" for name, value in metrics.items())
    return "\n".join(lines)


def _measure_multiline_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    spacing: int,
) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
    )
    return right - left, bottom - top


def _draw_centered_multiline_text(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> None:
    box_left, box_top, box_right, box_bottom = box
    text_width, text_height = _measure_multiline_text(
        draw,
        text,
        font=font,
        spacing=spacing,
    )
    text_x = box_left + max(0, (box_right - box_left - text_width) // 2)
    text_y = box_top + max(0, (box_bottom - box_top - text_height) // 2)
    draw.multiline_text(
        (text_x, text_y),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _resolve_visualization_path(
    config: Mapping[str, Any],
    output_path: str | Path | None,
) -> Path:
    if output_path is not None:
        return Path(output_path)

    output_cfg = dict(config.get("output", {}))
    output_dir = Path(output_cfg.get("dir", "./outputs"))
    return output_dir / "visualizations" / "comparison_grid.png"


def _select_sample_indices(
    dataset_size: int,
    *,
    num_samples: int,
    seed: int | None,
) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("Evaluation dataset is empty.")

    sample_count = min(num_samples, dataset_size)
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), sample_count))


def _build_bicubic_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    data_cfg = dict(config.get("data", {}))
    model = build_model("bicubic", scale_factor=int(data_cfg["scale"]))
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def visualize_comparison_grid(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_samples: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    current_model, checkpoint, device = load_model_and_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
    )
    bicubic_model = _build_bicubic_model(config, device)
    dataloader = build_eval_dataloader(config)
    dataset = dataloader.dataset

    eval_cfg = dict(config.get("eval", {}))
    metric_names = tuple(eval_cfg.get("metric_names", ("psnr", "ssim")))
    if not metric_names:
        raise ValueError("Visualization metric list is empty.")
    resolved_seed = int(config.get("seed", 42)) if seed is None else seed

    selected_indices = _select_sample_indices(
        len(dataset),
        num_samples=num_samples,
        seed=resolved_seed,
    )
    samples = [dataset[index] for index in selected_indices]

    first_hr = samples[0]["hr"]
    if not isinstance(first_hr, Tensor):
        raise TypeError("Expected dataset sample to contain tensor entry 'hr'.")

    cell_width = int(first_hr.shape[-1])
    cell_height = int(first_hr.shape[-2])
    header_height = 72
    outer_padding = 20
    row_gap = 16
    column_gap = 12
    text_spacing = 3

    current_model_cfg = dict(config.get("model", {}))
    current_model_name = str(current_model_cfg.get("kind", "model")).capitalize()

    font = ImageFont.load_default()
    canvas_width = outer_padding * 2 + cell_width * 3 + column_gap * 2
    row_height = header_height + cell_height
    canvas_height = outer_padding * 2 + row_height * len(samples) + row_gap * (len(samples) - 1)

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    per_image_results: list[dict[str, Any]] = []
    non_blocking = device.type == "cuda"

    for row_index, sample in enumerate(samples):
        lr = sample["lr"]
        hr = sample["hr"]
        image_id = str(sample["image_id"])

        if not isinstance(lr, Tensor) or not isinstance(hr, Tensor):
            raise TypeError("Expected dataset sample to contain tensor entries 'lr' and 'hr'.")

        lr_batch = lr.unsqueeze(0).to(device=device, non_blocking=non_blocking)
        hr_batch = hr.unsqueeze(0).to(device=device, non_blocking=non_blocking)

        bicubic_prediction = bicubic_model(lr_batch).detach().float().clamp_(0.0, 1.0)
        current_prediction = current_model(lr_batch).detach().float().clamp_(0.0, 1.0)
        target = hr_batch.detach().float()
        bicubic_prediction, target = align_image_channels(bicubic_prediction, target)
        current_prediction, target = align_image_channels(current_prediction, target)

        bicubic_metrics = compute_metrics(
            bicubic_prediction,
            target,
            metric_names=metric_names,
        )
        current_metrics = compute_metrics(
            current_prediction,
            target,
            metric_names=metric_names,
        )

        row_top = outer_padding + row_index * (row_height + row_gap)
        image_top = row_top + header_height

        cell_specs = [
            (
                _fit_image_to_cell(_tensor_to_pil_image(bicubic_prediction[0]), cell_size=(cell_width, cell_height)),
                _format_metric_lines("Bicubic", bicubic_metrics),
            ),
            (
                _fit_image_to_cell(_tensor_to_pil_image(current_prediction[0]), cell_size=(cell_width, cell_height)),
                _format_metric_lines(current_model_name, current_metrics),
            ),
            (
                _fit_image_to_cell(_tensor_to_pil_image(target[0]), cell_size=(cell_width, cell_height)),
                _format_metric_lines("HR", image_id=image_id),
            ),
        ]

        for column_index, (cell_image, cell_text) in enumerate(cell_specs):
            cell_left = outer_padding + column_index * (cell_width + column_gap)
            text_box = (
                cell_left,
                row_top,
                cell_left + cell_width,
                row_top + header_height,
            )
            image_box = (cell_left, image_top)

            _draw_centered_multiline_text(
                draw,
                box=text_box,
                text=cell_text,
                font=font,
                fill=(0, 0, 0),
                spacing=text_spacing,
            )
            canvas.paste(cell_image, image_box)

        per_image_results.append(
            {
                "image_id": image_id,
                "bicubic_metrics": bicubic_metrics,
                "model_metrics": current_metrics,
            }
        )

    resolved_output_path = _resolve_visualization_path(config, output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_output_path)

    print(
        f"Saved comparison grid with {len(samples)} samples to: {resolved_output_path}"
    )

    return {
        "output_path": str(resolved_output_path),
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "num_samples": len(samples),
        "metric_names": list(metric_names),
        "seed": resolved_seed,
        "samples": per_image_results,
    }
