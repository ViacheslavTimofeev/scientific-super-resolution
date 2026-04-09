from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.models.loading import load_model_and_checkpoint


def _resolve_output_path(
    config: Mapping[str, Any],
) -> Path:
    export_cfg = dict(config.get("export", {}))
    model_path = export_cfg.get("model_path")
    if model_path is None:
        raise KeyError("Export config must contain 'export.model_path'.")
    return Path(model_path)


def _resolve_input_tensor(
    config: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    export_cfg = dict(config.get("export", {}))
    configured_shape = export_cfg.get("input_shape")
    if configured_shape is None:
        raise KeyError("Export config must contain 'export.input_shape' with 4 dimensions.")
    resolved_tensor = tuple(int(value) for value in configured_shape)

    if len(resolved_tensor) != 4:
        raise ValueError(
            "ONNX export input_shape must contain exactly 4 dimensions: "
            "(batch, channels, height, width)."
        )

    if resolved_tensor[2] <= 0 or resolved_tensor[3] <= 0:
        raise ValueError("ONNX export input_shape must have positive height and width.")

    return resolved_tensor


def _resolve_checkpoint_path(config: Mapping[str, Any]) -> str | Path | None:
    export_cfg = dict(config.get("export", {}))
    checkpoint_path = export_cfg.get("checkpoint_path")
    if checkpoint_path is None:
        return None
    return Path(checkpoint_path)


def _resolve_verify_export(config: Mapping[str, Any]) -> bool:
    export_cfg = dict(config.get("export", {}))
    return bool(export_cfg.get("verify_export", True))


def _verify_exported_model(
    model_path: Path,
    *,
    dummy_input: torch.Tensor,
    pytorch_output: torch.Tensor,
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
) -> dict[str, Any]:
    resolved_checkpoint_path = _resolve_checkpoint_path(config)
    model, checkpoint, resolved_device = load_model_and_checkpoint(
        config,
        checkpoint_path=resolved_checkpoint_path,
    )
    resolved_input_shape = _resolve_input_tensor(config)
    resolved_output_path = _resolve_output_path(config)
    resolved_verify_export = _resolve_verify_export(config)

    dummy_input = torch.randn(*resolved_input_shape, device=resolved_device)
    model.eval()
    pytorch_output = model(dummy_input)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(resolved_output_path),
        export_params=True,
        opset_version=18,
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
        "opset_version": 18,
    }

    if resolved_verify_export:
        results["verification"] = _verify_exported_model(
            resolved_output_path,
            dummy_input=dummy_input,
            pytorch_output=pytorch_output,
        )

    return results
