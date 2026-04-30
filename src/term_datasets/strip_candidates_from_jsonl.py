from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed"
DEFAULT_SOURCE_FILENAMES = (
    "train_t1_candidates.jsonl",
    "test1_t1_candidates.jsonl",
    "test1_t3_candidates.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create CL-RuTerm3 jsonl files without the precomputed `candidates` field."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    processed_dir = args.processed_dir.expanduser().resolve()

    for source_name in DEFAULT_SOURCE_FILENAMES:
        source_path = processed_dir / source_name
        target_path = processed_dir / source_name.replace("_candidates.jsonl", "_nocands.jsonl")
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with source_path.open("r", encoding="utf-8-sig") as source, target_path.open("w", encoding="utf-8") as target:
            for line in source:
                row = json.loads(line)
                row.pop("candidates", None)
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved {target_path}")


if __name__ == "__main__":
    main()
