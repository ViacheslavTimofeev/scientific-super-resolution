from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export.export_onnx import export_onnx
from src.train.trainer import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a super-resolution model to ONNX.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, the best checkpoint from config is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional ONNX output path. If omitted, export.output_path or outputs/<experiment>.onnx is used.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=None,
        metavar=("B", "C", "H", "W"),
        help="Optional dummy input shape for export: batch channels height width.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional export device override.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset override.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip optional ONNX verification.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional path to save export results as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results = export_onnx(
        config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=args.input_shape,
        device=args.device,
        opset_version=args.opset,
        verify_export=not args.no_verify,
        save_results_path=args.save_results,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
