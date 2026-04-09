from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export.export_onnx import export_onnx
from src.models.loading import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a super-resolution model to ONNX.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results = export_onnx(config)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
