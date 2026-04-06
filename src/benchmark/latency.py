from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.train.loops import resolve_device


def _load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {resolved_checkpoint_path}"
        )

    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a mapping. "
            f"Got {type(checkpoint)!r} from '{resolved_checkpoint_path}'."
        )

    checkpoint["checkpoint_path"] = str(resolved_checkpoint_path)
    return checkpoint


def _prepare_model(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    device: str | torch.device | None,
    config: Mapping[str, Any] | None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    resolved_device = resolve_device(device, config)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise KeyError(
            "Checkpoint must contain 'model_state_dict' or be a raw state dict mapping."
        )

    model.load_state_dict(dict(state_dict))
    model.eval()
    model.to(resolved_device)
    return model, checkpoint, resolved_device


def _resolve_dtype(
    dtype: torch.dtype | str | None,
    *,
    device: torch.device,
) -> torch.dtype:
    if dtype is None:
        return torch.float32

    if isinstance(dtype, torch.dtype):
        return dtype

    normalized_dtype = str(dtype).lower()
    if normalized_dtype in {"float32", "fp32"}:
        return torch.float32
    if normalized_dtype in {"float16", "fp16", "half"}:
        if device.type != "cuda":
            raise ValueError("float16 latency benchmark is supported only on CUDA.")
        return torch.float16

    raise ValueError("Unsupported dtype. Use float32/fp32 or float16/fp16.")


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_forward_pass(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> list[float]:
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(iterations):
            _synchronize_if_needed(device)
            start = time.perf_counter()
            _ = model(sample_input)
            _synchronize_if_needed(device)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    return latencies_ms


def _save_json(data: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, indent=2, ensure_ascii=False)


def measure_latency(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    input_shape: tuple[int, ...],
    device: str | torch.device | None,
    config: Mapping[str, Any] | None = None,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    dtype: torch.dtype | str | None = None,
    save_results_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Measure model inference latency after checkpoint loading and warmup.
    """
    if len(input_shape) != 4:
        raise ValueError(
            "input_shape must contain exactly 4 dimensions: (batch, channels, height, width)."
        )
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be >= 0.")
    if benchmark_iters <= 0:
        raise ValueError("benchmark_iters must be > 0.")

    model, checkpoint, resolved_device = _prepare_model(
        model,
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
    )
    resolved_dtype = _resolve_dtype(dtype, device=resolved_device)

    sample_input = torch.randn(
        *input_shape,
        device=resolved_device,
        dtype=resolved_dtype,
    )

    if warmup_iters > 0:
        _run_forward_pass(
            model,
            sample_input,
            iterations=warmup_iters,
            device=resolved_device,
        )

    latencies_ms = _run_forward_pass(
        model,
        sample_input,
        iterations=benchmark_iters,
        device=resolved_device,
    )

    results = {
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "device": str(resolved_device),
        "dtype": str(resolved_dtype).replace("torch.", ""),
        "input_shape": list(input_shape),
        "warmup_iters": warmup_iters,
        "benchmark_iters": benchmark_iters,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "std": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        },
        "throughput_fps": 1000.0 / statistics.fmean(latencies_ms),
    }
    if save_results_path is not None:
        _save_json(results, save_results_path)
        results["results_path"] = str(Path(save_results_path))

    return results
