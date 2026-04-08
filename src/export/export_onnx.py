from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.models.loading import load_model_and_checkpoint


def _resolve_output_path(
    config: Mapping[str, Any],
    output_path: str | Path | None,
) -> Path:
    if output_path is not None:
        return Path(output_path)

    export_cfg = dict(config.get("export", {}))
    configured_path = export_cfg.get("output_path")
    if configured_path is not None:
        return Path(configured_path)

    output_cfg = dict(config.get("output", {}))
    output_dir = Path(output_cfg.get("dir", "./outputs"))
    experiment_name = str(config.get("experiment_name", "model"))
    return output_dir / f"{experiment_name}.onnx"


def _resolve_input_channels(config: Mapping[str, Any]) -> int:
    model_cfg = dict(config.get("model", {}))
    if "in_channels" in model_cfg:
        return int(model_cfg["in_channels"])
    if "in_chans" in model_cfg:
        return int(model_cfg["in_chans"])

    data_cfg = dict(config.get("data", {}))
    image_mode = str(data_cfg.get("image_mode", "RGB")).upper()
    return 1 if image_mode == "L" else 3


def _resolve_spatial_size(config: Mapping[str, Any]) -> tuple[int, int]:
    model_cfg = dict(config.get("model", {}))
    img_size = model_cfg.get("img_size")
    if isinstance(img_size, int):
        return img_size, img_size
    if (
        isinstance(img_size, Sequence)
        and len(img_size) == 2
        and all(isinstance(value, int) for value in img_size)
    ):
        return int(img_size[0]), int(img_size[1])

    data_cfg = dict(config.get("data", {}))
    patch_size = data_cfg.get("patch_size")
    scale = data_cfg.get("scale")
    if isinstance(patch_size, int) and isinstance(scale, int) and scale > 0:
        lr_size = patch_size // scale
        if lr_size > 0:
            return lr_size, lr_size

    return 64, 64


def _resolve_input_shape(
    config: Mapping[str, Any],
    input_shape: Sequence[int] | None,
) -> tuple[int, int, int, int]:
    if input_shape is not None:
        resolved_shape = tuple(int(value) for value in input_shape)
    else:
        export_cfg = dict(config.get("export", {}))
        configured_shape = export_cfg.get("input_shape")
        if configured_shape is not None:
            resolved_shape = tuple(int(value) for value in configured_shape)
        else:
            channels = _resolve_input_channels(config)
            height, width = _resolve_spatial_size(config)
            resolved_shape = (1, channels, height, width)

    if len(resolved_shape) != 4:
        raise ValueError(
            "ONNX export input_shape must contain exactly 4 dimensions: "
            "(batch, channels, height, width)."
        )

    return resolved_shape


def _resolve_opset_version(config: Mapping[str, Any], opset_version: int | None) -> int:
    if opset_version is not None:
        return int(opset_version)

    export_cfg = dict(config.get("export", {}))
    return int(export_cfg.get("opset_version", 17))


def _resolve_verify_export(config: Mapping[str, Any], verify_export: bool | None) -> bool:
    if verify_export is not None:
        return bool(verify_export)

    export_cfg = dict(config.get("export", {}))
    return bool(export_cfg.get("verify_export", True))


def _save_json(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, indent=2, ensure_ascii=False)


def _verify_exported_model(
    model_path: Path,
    *,
    dummy_input: torch.Tensor | None = None,
    pytorch_output: torch.Tensor | None = None,
) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort

    verification: dict[str, Any] = {
        "onnx_checker": None,
        "onnxruntime": None,
    }

    loaded_model = onnx.load(str(model_path))
    onnx.checker.check_model(loaded_model)
    verification["onnx_checker"] = {
        "available": True,
        "passed": True,
    }

    if dummy_input is None or pytorch_output is None:
        return verification

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    expected_output = pytorch_output.detach().cpu().numpy()
    runtime_outputs = session.run(
        None,
        {input_name: dummy_input.detach().cpu().numpy()},
    )
    runtime_output = runtime_outputs[0]
    verification["onnxruntime"] = {
        "available": True,
        "passed": bool(np.allclose(runtime_output, expected_output, rtol=1e-3, atol=1e-4)),
        "max_abs_diff": float(np.max(np.abs(runtime_output - expected_output))),
    }
    return verification


@torch.inference_mode()
def export_onnx(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    input_shape: Sequence[int] | None = None,
    device: str | torch.device | None = None,
    opset_version: int | None = None,
    verify_export: bool | None = None,
    save_results_path: str | Path | None = None,
) -> dict[str, Any]:
    model, checkpoint, resolved_device = load_model_and_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    resolved_input_shape = _resolve_input_shape(config, input_shape)
    resolved_output_path = _resolve_output_path(config, output_path)
    resolved_opset_version = _resolve_opset_version(config, opset_version)
    resolved_verify_export = _resolve_verify_export(config, verify_export)

    dummy_input = torch.randn(*resolved_input_shape, device=resolved_device)
    model.eval()
    pytorch_output = model(dummy_input)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(resolved_output_path),
        export_params=True,
        opset_version=resolved_opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "onnx_path": str(resolved_output_path),
        "device": str(resolved_device),
        "input_shape": list(resolved_input_shape),
        "opset_version": resolved_opset_version,
    }

    if resolved_verify_export:
        results["verification"] = _verify_exported_model(
            resolved_output_path,
            dummy_input=dummy_input,
            pytorch_output=pytorch_output,
        )

    if save_results_path is not None:
        results_path = Path(save_results_path)
        _save_json(results, results_path)
        results["results_path"] = str(results_path)

    return results
