from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from io import BytesIO
from pathlib import Path
import sys
from typing import Any
from urllib import error, parse, request

import gradio as gr
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loading import load_config


DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_inference.yaml"


def _load_demo_config() -> dict[str, Any]:
    config = load_config(DEMO_CONFIG_PATH)
    app_cfg = dict(config.get("app", {}))
    return {"app": app_cfg}


def _get_app_config() -> dict[str, Any]:
    return dict(_load_demo_config().get("app", {}))


def _resolve_api_url(args: argparse.Namespace) -> str:
    if args.api_url:
        return args.api_url.rstrip("/")

    app_cfg = _get_app_config()
    api_host = str(app_cfg.get("api_host", "127.0.0.1"))
    api_port = int(app_cfg.get("api_port", 8000))
    return f"http://{api_host}:{api_port}"


def _fetch_json(url: str) -> Any:
    try:
        with request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach API at {url}: {exc.reason}") from exc


def _build_multipart_form_data(image_bytes: bytes, filename: str) -> tuple[bytes, str]:
    boundary = f"sr-demo-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(image_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), boundary


def _predict_via_api(api_url: str, image: Image.Image, preset_name: str) -> tuple[Image.Image, dict[str, str]]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    body, boundary = _build_multipart_form_data(buffer.getvalue(), "input.png")
    query = parse.urlencode({"preset": preset_name})
    predict_url = f"{api_url}/predict?{query}"
    api_request = request.Request(
        predict_url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with request.urlopen(api_request) as response:
            image_bytes = response.read()
            headers = {key: value for key, value in response.headers.items()}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Prediction failed with {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach API at {predict_url}: {exc.reason}") from exc

    with Image.open(BytesIO(image_bytes)) as output_image:
        return output_image.copy(), headers


def _make_bicubic_preview(image: Image.Image, scale: int) -> Image.Image:
    return image.resize(
        (image.width * scale, image.height * scale),
        resample=Image.Resampling.BICUBIC,
    )


def _normalize_input_preview(image: Image.Image, output_mode: str) -> Image.Image:
    if output_mode in {"L", "RGB"}:
        converted = image.convert(output_mode)
    else:
        converted = image.copy()
    return converted.convert("RGB")


def _load_presets(api_url: str) -> list[dict[str, Any]]:
    payload = _fetch_json(f"{api_url}/presets")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("API returned no presets.")
    return payload


def _get_header(headers: dict[str, str], key: str, default: str) -> str:
    lowered = key.lower()
    for header_name, header_value in headers.items():
        if header_name.lower() == lowered:
            return header_value
    return default


def _format_summary(
    *,
    preset_name: str,
    preset_info: dict[str, Any],
    headers: dict[str, str],
    api_url: str,
) -> str:
    summary_lines = [
        f"Frontend: Gradio demo",
        f"Backend: {api_url}",
        f"Preset: {preset_name}",
        f"Model kind: {_get_header(headers, 'X-SR-Model-Kind', str(preset_info.get('model_kind', 'unknown')))}",
        f"Device: {_get_header(headers, 'X-SR-Device', str(preset_info.get('device', 'unknown')))}",
        f"Scale: x{_get_header(headers, 'X-SR-Scale', str(preset_info.get('scale', 'unknown')))}",
        f"Input size: {_get_header(headers, 'X-SR-Input-Size', 'unknown')}",
        f"Output size: {_get_header(headers, 'X-SR-Output-Size', 'unknown')}",
        f"Channels: {preset_info.get('channels', 'unknown')}",
        f"Image mode: {preset_info.get('image_mode', 'unknown')}",
    ]

    crop_note = _get_header(headers, "X-SR-Preprocess-Note", "")
    if crop_note:
        summary_lines.append(crop_note)

    return "\n".join(summary_lines)


def run_super_resolution(
    image: Image.Image | None,
    preset_name: str,
    api_url: str,
    preset_catalog: list[dict[str, Any]],
) -> tuple[list[tuple[Image.Image, str]], str]:
    if image is None:
        raise gr.Error("Upload an image first.")

    preset_info = next((item for item in preset_catalog if item["name"] == preset_name), None)
    if preset_info is None:
        raise gr.Error(f"Unknown preset: {preset_name}")

    try:
        sr_image, headers = _predict_via_api(api_url, image, preset_name)
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    scale = int(_get_header(headers, "X-SR-Scale", str(preset_info["scale"])))
    input_preview = _normalize_input_preview(image, str(preset_info.get("image_mode", "RGB")))
    bicubic_preview = _make_bicubic_preview(input_preview, scale)
    sr_preview = sr_image.convert("RGB")

    gallery = [
        (input_preview, "LR input"),
        (bicubic_preview, f"Bicubic x{scale}"),
        (sr_preview, f"{preset_name} SR"),
    ]
    summary = _format_summary(
        preset_name=preset_name,
        preset_info=preset_info,
        headers=headers,
        api_url=api_url,
    )
    return gallery, summary


def build_demo(api_url: str) -> gr.Blocks:
    app_cfg = _get_app_config()
    presets = _load_presets(api_url)
    preset_names = [str(item["name"]) for item in presets]

    title = str(app_cfg.get("title", "SR Demo"))
    description = str(
        app_cfg.get(
            "description",
            "Upload a low-resolution image and compare frontend previews with backend SR inference.",
        )
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{description}")
        gr.Markdown(f"Backend API: `{api_url}`")

        preset_catalog_state = gr.State(presets)
        api_url_state = gr.State(api_url)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload LR image")
                preset_input = gr.Dropdown(
                    choices=preset_names,
                    value=preset_names[0],
                    label="Model preset",
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
                summary_output = gr.Textbox(label="Run summary", lines=10)

        run_button.click(
            fn=run_super_resolution,
            inputs=[image_input, preset_input, api_url_state, preset_catalog_state],
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
    parser.add_argument(
        "--api-url",
        default=str(app_cfg.get("api_url", "")),
        help="Base URL of the inference API backend, for example http://127.0.0.1:8000.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_url = _resolve_api_url(args)
    demo = build_demo(api_url)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
