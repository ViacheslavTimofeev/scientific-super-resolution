from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from src.models.factory import build_model
from src.models.loading import extract_state_dict, load_checkpoint, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_inference.yaml"


@dataclass(slots=True)
class LoadedPreset:
    name: str
    model: torch.nn.Module
    device: torch.device
    scale: int
    image_mode: str
    crop_multiple: int | None
    channels: int
    model_kind: str


class InferenceService:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._config: dict[str, Any] = {}
        self._presets: dict[str, LoadedPreset] = {}

    def load(self) -> None:
        config = load_config(self._config_path)
        presets = config.get("presets", {})
        if not isinstance(presets, dict) or not presets:
            raise ValueError("API config must contain a non-empty 'presets' mapping.")

        requested_device = self._resolve_default_device(config)
        loaded_presets: dict[str, LoadedPreset] = {}

        for preset_name, preset_spec in presets.items():
            if not isinstance(preset_spec, dict):
                raise ValueError(f"Preset '{preset_name}' must be a mapping.")

            checkpoint_path = PROJECT_ROOT / str(preset_spec["checkpoint"])
            if not checkpoint_path.exists():
                continue

            model_cfg = dict(preset_spec.get("model", {}))
            if "kind" not in model_cfg:
                raise KeyError(f"Preset '{preset_name}' is missing model.kind.")

            model_kind = str(model_cfg.pop("kind"))
            model = build_model(model_kind, **model_cfg)
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = extract_state_dict(checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(requested_device)

            loaded_presets[str(preset_name)] = LoadedPreset(
                name=str(preset_name),
                model=model,
                device=requested_device,
                scale=int(preset_spec["scale"]),
                image_mode=self._resolve_image_mode(preset_spec),
                crop_multiple=self._resolve_crop_multiple(preset_spec.get("crop_multiple")),
                channels=self._resolve_model_channels(preset_spec),
                model_kind=model_kind,
            )

        if not loaded_presets:
            raise RuntimeError(
                f"No API presets could be loaded. Check checkpoint paths in {self._config_path}."
            )

        self._config = config
        self._presets = loaded_presets

    @property
    def presets(self) -> dict[str, LoadedPreset]:
        return self._presets

    @property
    def default_preset_name(self) -> str:
        return next(iter(self._presets))

    def predict(self, image_bytes: bytes, preset_name: str) -> tuple[bytes, dict[str, str]]:
        preset = self._presets.get(preset_name)
        if preset is None:
            raise KeyError(f"Unknown preset '{preset_name}'.")

        image = self._read_image(image_bytes)
        prepared_image = image.convert(preset.image_mode)
        prepared_image, crop_note = self._center_crop_to_multiple(
            prepared_image,
            multiple=preset.crop_multiple,
        )

        input_tensor = tv_to_tensor(prepared_image).unsqueeze(0).to(preset.device)
        with torch.inference_mode():
            prediction = preset.model(input_tensor)[0]

        output_image = self._tensor_to_pil_image(prediction)
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")

        headers = {
            "X-SR-Preset": preset.name,
            "X-SR-Scale": str(preset.scale),
            "X-SR-Model-Kind": preset.model_kind,
            "X-SR-Device": str(preset.device),
            "X-SR-Input-Size": f"{prepared_image.width}x{prepared_image.height}",
            "X-SR-Output-Size": f"{output_image.width}x{output_image.height}",
        }
        if crop_note is not None:
            headers["X-SR-Preprocess-Note"] = crop_note

        return buffer.getvalue(), headers

    @staticmethod
    def _read_image(image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.copy()
        except UnidentifiedImageError as error:
            raise ValueError("Uploaded file is not a valid image.") from error

    @staticmethod
    def _resolve_default_device(config: dict[str, Any]) -> torch.device:
        app_cfg = dict(config.get("app", {}))
        requested = str(app_cfg.get("default_device", "auto")).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return device

    @staticmethod
    def _resolve_model_channels(preset: dict[str, Any]) -> int:
        model_cfg = dict(preset.get("model", {}))
        if "in_channels" in model_cfg:
            return int(model_cfg["in_channels"])
        if "in_chans" in model_cfg:
            return int(model_cfg["in_chans"])
        return 1

    @staticmethod
    def _resolve_image_mode(preset: dict[str, Any]) -> str:
        image_mode = str(preset.get("image_mode", "L"))
        if image_mode in {"L", "RGB"}:
            return image_mode
        return "RGB" if InferenceService._resolve_model_channels(preset) == 3 else "L"

    @staticmethod
    def _resolve_crop_multiple(value: Any) -> int | None:
        if value is None:
            return None
        crop_multiple = int(value)
        return crop_multiple if crop_multiple > 1 else None

    @staticmethod
    def _center_crop_to_multiple(
        image: Image.Image,
        *,
        multiple: int | None,
    ) -> tuple[Image.Image, str | None]:
        if multiple is None:
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

    @staticmethod
    def _tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
        image = image.detach().float().clamp_(0.0, 1.0).cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}.")

        if image.shape[0] == 1:
            array = (image.squeeze(0) * 255.0).round().to(torch.uint8).numpy()
            return Image.fromarray(array, mode="L")

        if image.shape[0] == 3:
            array = (image.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
            return Image.fromarray(array, mode="RGB")

        raise ValueError(f"Unsupported number of channels: {image.shape[0]}.")


def create_app(config_path: Path | None = None) -> FastAPI:
    service = InferenceService(config_path or DEFAULT_API_CONFIG_PATH)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.load()
        yield

    app = FastAPI(
        title="Super-Resolution Inference API",
        description="Backend inference service for image super-resolution models.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def healthcheck() -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded_presets": list(service.presets),
            "default_preset": service.default_preset_name if service.presets else None,
        }

    @app.get("/presets")
    async def list_presets() -> JSONResponse:
        payload = [
            {
                "name": preset.name,
                "scale": preset.scale,
                "device": str(preset.device),
                "image_mode": preset.image_mode,
                "channels": preset.channels,
                "model_kind": preset.model_kind,
            }
            for preset in service.presets.values()
        ]
        return JSONResponse(content=payload)

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        preset: str = Query(default=""),
    ) -> Response:
        selected_preset = preset or service.default_preset_name

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            output_bytes, headers = service.predict(image_bytes, selected_preset)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        return Response(content=output_bytes, media_type="image/png", headers=headers)

    return app


app = create_app()
