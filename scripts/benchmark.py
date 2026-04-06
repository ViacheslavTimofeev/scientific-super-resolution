from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.latency import measure_latency
from src.models.factory import build_model
from src.train.trainer import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model inference latency.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML benchmark config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, the config path is used.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional path to save benchmark results as JSON.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(
    config: dict[str, Any],
    checkpoint_path: str | None,
) -> str:
    if checkpoint_path is not None:
        return checkpoint_path

    benchmark_cfg = dict(config.get("benchmark", {}))
    checkpoint_from_cfg = benchmark_cfg.get("checkpoint_path")
    if checkpoint_from_cfg is None:
        raise KeyError(
            "Benchmark config must contain 'benchmark.checkpoint_path' "
            "or it must be provided via --checkpoint."
        )

    return str(checkpoint_from_cfg)


def _build_model_from_config(config: dict[str, Any]):
    model_cfg = dict(config.get("model", {}))
    model_kind = model_cfg.pop("kind", None)
    if model_kind is None:
        raise KeyError("Model config must contain 'kind'.")

    return build_model(str(model_kind), **model_cfg)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    benchmark_cfg = dict(config.get("benchmark", {}))

    input_shape = tuple(benchmark_cfg.get("input_shape", ()))
    if not input_shape:
        raise KeyError("Benchmark config must contain 'benchmark.input_shape'.")

    model = _build_model_from_config(config)
    checkpoint_path = _resolve_checkpoint_path(config, args.checkpoint)
    save_results_path = args.save_results or benchmark_cfg.get("save_results_path")

    results = measure_latency(
        model,
        checkpoint_path=checkpoint_path,
        input_shape=input_shape,
        device=benchmark_cfg.get("device", config.get("device")),
        config=config,
        warmup_iters=int(benchmark_cfg.get("warmup_iters", 10)),
        benchmark_iters=int(benchmark_cfg.get("benchmark_iters", 100)),
        dtype=benchmark_cfg.get("dtype"),
        save_results_path=save_results_path,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
