from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.visualize import visualize_comparison_grid
from src.train.trainer import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a comparison grid for bicubic, model prediction, and HR."
    )
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
        help="Optional output image path. If omitted, the default visualization path is used.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random evaluation samples to include in the grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for sample selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = visualize_comparison_grid(
        config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
