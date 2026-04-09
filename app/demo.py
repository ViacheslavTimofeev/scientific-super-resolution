from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor as tv_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model
from src.models.loading import load_config, load_model_and_checkpoint


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "CNN x2": {
        "config": "configs/train_cnn.yaml",
        "checkpoint": "outputs/cnn_x2_baseline/checkpoints/best.pt",
        "overrides": {
            "experiment_name": "cnn_x2_demo",
            "device": "cpu",
            "data": {
                "scale": 2,
            },
            "model": {
                "upscale": 2,
            },
        },
    },
    "CNN x4": {
        "config": "configs/benchmark_cnn.yaml",
        "checkpoint": "outputs/cnn_x4_baseline/checkpoints/best.pt",
    },
    "SwinIR x2": {
        "config": "configs/train_swinir.yaml",
        "checkpoint": "outputs/swinir_x2_baseline/checkpoints/best.pth",
    },
    "SwinIR x4": {
        "config": "configs/benchmark_swinir.yaml",
        "checkpoint": "outputs/swinir_x4_baseline/checkpoints/best.pth",
    },
}


def _available_presets() -> list[str]:
    available: list[str] = []
    for preset_name, preset in MODEL_PRESETS.items():
        config_path = PROJECT_ROOT / preset["config"]
        checkpoint_path = PROJECT_ROOT / preset["checkpoint"]
        if config_path.exists() and checkpoint_path.exists():
            available.append(preset_name)
    return available


def _resolve_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_channels(model_cfg: dict[str, Any]) -> int:
    if "in_channels" in model_cfg:
        return int(model_cfg["in_channels"])
    if "in_chans" in model_cfg:
        return int(model_cfg["in_chans"])
    return 1


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_scale(config: dict[str, Any]) -> int:
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    if "scale" in data_cfg:
        return int(data_cfg["scale"])
    if "upscale" in model_cfg:
        return int(model_cfg["upscale"])
    raise KeyError("Unable to resolve scale from config.")


def _resolve_crop_multiple(config: dict[str, Any]) -> int | None:
    model_cfg = dict(config.get("model", {}))
    if str(model_cfg.get("kind", "")).lower() == "swinir":
        return int(model_cfg.get("window_size", 1))
    return None


def _center_crop_to_multiple(
    image: Image.Image,
    *,
    multiple: int | None,
) -> tuple[Image.Image, str | None]:
    if multiple is None or multiple <= 1:
        return image, None

    width, height = image.size
    cropped_width = (width // multiple) * multiple
    cropped_height = (height // multiple) * multiple

    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(
            f"Image size {(width, height)} is too small for required multiple={multiple}."
        )

    if cropped_width == width and cropped_height == height:
        return image, None

    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    cropped = image.crop((left, top, left + cropped_width, top + cropped_height))
    note = (
        f"Input was center-cropped from {width}x{height} to "
        f"{cropped_width}x{cropped_height} to match model constraints."
    )
    return cropped, note


def _tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().clamp_(0.0, 1.0).cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}.")

    if image.shape[0] == 1:
        array = (image.squeeze(0) * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(array, mode="L").convert("RGB")

    if image.shape[0] == 3:
        array = (image.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(array, mode="RGB")

    raise ValueError(f"Unsupported number of channels: {image.shape[0]}.")


@lru_cache(maxsize=8)
def _load_predictor(
    preset_name: str,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    preset = MODEL_PRESETS[preset_name]
    config_path = PROJECT_ROOT / preset["config"]
    checkpoint_path = PROJECT_ROOT / preset["checkpoint"]

    config = load_config(config_path)
    overrides = preset.get("overrides")
    if isinstance(overrides, dict):
        config = _deep_update(config, overrides)
    config["device"] = device
    model, _, resolved_device = load_model_and_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return model, config, resolved_device


@torch.inference_mode()
def run_super_resolution(
    image: Image.Image | None,
    preset_name: str,
    device: str,
) -> tuple[list[tuple[Image.Image, str]], str]:
    if image is None:
        raise gr.Error("Upload an image first.")

    if preset_name not in MODEL_PRESETS:
        raise gr.Error(f"Unknown preset: {preset_name}")

    model, config, resolved_device = _load_predictor(preset_name, device)
    model_cfg = dict(config.get("model", {}))

    channels = _resolve_model_channels(model_cfg)
    image_mode = "RGB" if channels == 3 else "L"
    prepared_image = image.convert(image_mode)
    prepared_image, crop_note = _center_crop_to_multiple(
        prepared_image,
        multiple=_resolve_crop_multiple(config),
    )

    input_tensor = tv_to_tensor(prepared_image).unsqueeze(0).to(resolved_device)
    scale = _resolve_scale(config)

    bicubic_model = build_model("bicubic", scale_factor=scale).to(resolved_device).eval()
    bicubic_prediction = bicubic_model(input_tensor)[0]
    sr_prediction = model(input_tensor)[0]

    gallery = [
        (_tensor_to_pil_image(input_tensor[0]), "LR input"),
        (_tensor_to_pil_image(bicubic_prediction), f"Bicubic x{scale}"),
        (_tensor_to_pil_image(sr_prediction), f"{preset_name} SR"),
    ]

    summary_lines = [
        f"Preset: {preset_name}",
        f"Backend: direct PyTorch inference",
        f"Device: {resolved_device}",
        f"Input size: {prepared_image.width}x{prepared_image.height}",
        f"Output size: {prepared_image.width * scale}x{prepared_image.height * scale}",
        f"Channels: {channels}",
    ]
    if crop_note is not None:
        summary_lines.append(crop_note)

    return gallery, "\n".join(summary_lines)


def build_demo() -> gr.Blocks:
    presets = _available_presets()
    if not presets:
        raise RuntimeError(
            "No model presets are available. Expected configs in ./configs and checkpoints in ./outputs."
        )

    default_device = _resolve_default_device()

    with gr.Blocks(title="SR Demo") as demo:
        gr.Markdown(
            """
            # Super-Resolution Demo
            Upload a low-resolution image, choose a model preset, and compare LR, bicubic, and SR outputs.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload LR image",
                )
                preset_input = gr.Dropdown(
                    choices=presets,
                    value=presets[0],
                    label="Model preset",
                )
                device_input = gr.Dropdown(
                    choices=["cpu", "cuda"],
                    value=default_device,
                    label="Device",
                )
                run_button = gr.Button("Run super-resolution", variant="primary")

            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Comparison",
                    columns=3,
                    object_fit="contain",
                    preview=True,
                    height="auto",
                )
                summary_output = gr.Textbox(
                    label="Run summary",
                    lines=8,
                )

        run_button.click(
            fn=run_super_resolution,
            inputs=[image_input, preset_input, device_input],
            outputs=[output_gallery, summary_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the super-resolution demo UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the Gradio server.")
    parser.add_argument("--port", default=7860, type=int, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
