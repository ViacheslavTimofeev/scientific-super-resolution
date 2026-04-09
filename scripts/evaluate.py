from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluate import evaluate
from src.models.loading import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a super-resolution model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML evaluation config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint. If omitted, the best checkpoint from config is used.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional path to save aggregated and per-image results as JSON.",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Return and print per-image results in the CLI output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results = evaluate(
        config,
        checkpoint_path=args.checkpoint,
        return_per_image=args.per_image,
        save_results_path=args.save_results,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
