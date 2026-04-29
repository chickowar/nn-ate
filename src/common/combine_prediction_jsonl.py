from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_FILENAMES = (
    "test2_t3_v2.jsonl",
    "test2_t12_v2.jsonl",
    "test_all_domains_v2.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine fixed prediction jsonl files from a directory into one jsonl, "
            "keeping only id, text and label."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing test2_t3_v2.jsonl, test2_t12_v2.jsonl and test_all_domains_v2.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path for the combined jsonl. Defaults to <input_dir>/combined_predictions.jsonl.",
    )
    return parser


def normalize_label_item(raw_item: Any) -> list[Any]:
    if not isinstance(raw_item, list):
        raise ValueError(f"Label item must be a list, got: {raw_item!r}")

    if len(raw_item) == 2:
        start, end = raw_item
        return [int(start), int(end), "specific"]

    if len(raw_item) == 3:
        start, end, label_class = raw_item
        return [int(start), int(end), str(label_class)]

    raise ValueError(f"Label item must have length 2 or 3, got: {raw_item!r}")


def normalize_label(raw_label: Any) -> list[list[Any]]:
    if raw_label is None:
        return []
    if not isinstance(raw_label, list):
        raise ValueError(f"label must be a list, got: {raw_label!r}")
    return [normalize_label_item(item) for item in raw_label]


def normalize_row(payload: dict[str, Any]) -> dict[str, Any]:
    if "id" not in payload:
        raise ValueError(f"Missing 'id' in row: {payload!r}")
    if "text" not in payload:
        raise ValueError(f"Missing 'text' in row: {payload!r}")

    return {
        "id": payload["id"],
        "text": payload["text"],
        "label": normalize_label(payload.get("label")),
    }


def iter_normalized_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                rows.append(normalize_row(payload))
            except Exception as error:
                raise ValueError(f"Failed to parse {path} line {line_number}: {error}") from error
    return rows


def resolve_input_paths(input_dir: Path) -> list[Path]:
    paths = [input_dir / filename for filename in EXPECTED_FILENAMES]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files: {missing_str}")
    return paths


def combine_prediction_files(input_dir: Path, output_path: Path) -> int:
    input_paths = resolve_input_paths(input_dir)
    total_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for input_path in input_paths:
            for row in iter_normalized_rows(input_path):
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    return total_rows


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_path = (args.output_path or (input_dir / "combined_predictions.jsonl")).resolve()
    total_rows = combine_prediction_files(input_dir=input_dir, output_path=output_path)
    print(f"Wrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
