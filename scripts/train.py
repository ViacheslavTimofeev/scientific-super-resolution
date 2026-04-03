from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.trainer import load_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML training config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
