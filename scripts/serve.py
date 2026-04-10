from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import DEFAULT_API_CONFIG_PATH, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the super-resolution API server.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the API server.",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="Port for the API server.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_API_CONFIG_PATH),
        help="Path to the API config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(Path(args.config).resolve())
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
