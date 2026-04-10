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
from src.models.loading import extract_state_dict, load_checkpoint, load_config


DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_inference.yaml"


@lru_cache(maxsize=1)
def _load_demo_config() -> dict[str, Any]:
    config = load_config(DEMO_CONFIG_PATH)
    if "app" not in config or "presets" not in config:
        raise KeyError("Demo config must contain 'app' and 'presets' sections.")
    return config


def _get_app_config() -> dict[str, Any]:
    return dict(_load_demo_config().get("app", {}))


def _get_preset_specs() -> dict[str, dict[str, Any]]:
    presets = _load_demo_config().get("presets", {})
    if not isinstance(presets, dict) or not presets:
        raise ValueError("Demo config 'presets' must be a non-empty mapping.")
    return {str(name): dict(spec) for name, spec in presets.items()}


def _available_presets() -> list[str]:
    available: list[str] = []
    for preset_name, preset in _get_preset_specs().items():
        checkpoint_path = PROJECT_ROOT / str(preset["checkpoint"])
        if checkpoint_path.exists():
            available.append(preset_name)
    return available


def _resolve_default_device() -> str:
    requested = str(_get_app_config().get("default_device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_model_channels(preset: dict[str, Any]) -> int:
    model_cfg = dict(preset.get("model", {}))
    if "in_channels" in model_cfg:
        return int(model_cfg["in_channels"])
    if "in_chans" in model_cfg:
        return int(model_cfg["in_chans"])
    return 1


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
    preset = _get_preset_specs()[preset_name]
    checkpoint_path = PROJECT_ROOT / str(preset["checkpoint"])
    model_cfg = dict(preset.get("model", {}))
    model_kind = str(model_cfg.pop("kind"))

    config = {
        "device": device,
        "data": {
            "scale": int(preset["scale"]),
        },
        "model": {
            "kind": model_kind,
            **model_cfg,
        },
    }

    model = build_model(model_kind, **model_cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    model.to(resolved_device)
    return model, config, resolved_device


@torch.inference_mode()
def run_super_resolution(
    image: Image.Image | None,
    preset_name: str,
    device: str,
) -> tuple[list[tuple[Image.Image, str]], str]:
    if image is None:
        raise gr.Error("Upload an image first.")

    preset_specs = _get_preset_specs()
    if preset_name not in preset_specs:
        raise gr.Error(f"Unknown preset: {preset_name}")

    preset = preset_specs[preset_name]
    model, config, resolved_device = _load_predictor(preset_name, device)

    channels = _resolve_model_channels(preset)
    image_mode = str(preset.get("image_mode", "L"))
    if image_mode not in {"L", "RGB"}:
        image_mode = "RGB" if channels == 3 else "L"

    prepared_image = image.convert(image_mode)
    prepared_image, crop_note = _center_crop_to_multiple(
        prepared_image,
        multiple=preset.get("crop_multiple"),
    )

    input_tensor = tv_to_tensor(prepared_image).unsqueeze(0).to(resolved_device)
    scale = int(preset["scale"])

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
        f"Model kind: {preset['model']['kind']}",
        f"Input size: {prepared_image.width}x{prepared_image.height}",
        f"Output size: {prepared_image.width * scale}x{prepared_image.height * scale}",
        f"Channels: {channels}",
    ]
    if crop_note is not None:
        summary_lines.append(crop_note)

    return gallery, "\n".join(summary_lines)


def build_demo() -> gr.Blocks:
    app_cfg = _get_app_config()
    presets = _available_presets()
    if not presets:
        raise RuntimeError(
            "No demo presets are available. Check checkpoint paths in configs/demo_inference.yaml."
        )

    default_device = _resolve_default_device()
    title = str(app_cfg.get("title", "SR Demo"))
    description = str(
        app_cfg.get(
            "description",
            "Upload a low-resolution image, choose a model preset, and compare LR, bicubic, and SR outputs.",
        )
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{description}")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload LR image")
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
                summary_output = gr.Textbox(label="Run summary", lines=8)

        run_button.click(
            fn=run_super_resolution,
            inputs=[image_input, preset_input, device_input],
            outputs=[output_gallery, summary_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    app_cfg = _get_app_config()

    parser = argparse.ArgumentParser(description="Launch the super-resolution demo UI.")
    parser.add_argument(
        "--host",
        default=str(app_cfg.get("host", "127.0.0.1")),
        help="Host interface for the Gradio server.",
    )
    parser.add_argument(
        "--port",
        default=int(app_cfg.get("port", 7860)),
        type=int,
        help="Port for the Gradio server.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
