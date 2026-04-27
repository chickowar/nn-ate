from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.rutermeval_score import evaluate_track1, load_jsonl


DEFAULT_GOLD_PATH = Path("data/CL-RuTerm3/processed/test1_t1_candidates.jsonl")
DEFAULT_INFERENCE_DIRS = (
    Path("data/CL-RuTerm3/inference/less_old"),
    Path("data/CL-RuTerm3/inference/old"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute RuTermEval scores for all test inference files in configured directories."
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Path to gold jsonl file. Default: {DEFAULT_GOLD_PATH}",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        action="append",
        dest="inference_dirs",
        help="Inference directory to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--glob",
        default="test*.jsonl",
        help="Filename glob for prediction files inside each inference directory.",
    )
    parser.add_argument("--full-text-prefix", default="ft-")
    parser.add_argument("--strict-ids", action="store_true")
    return parser


def iter_prediction_files(inference_dirs: tuple[Path, ...], pattern: str) -> list[Path]:
    files: list[Path] = []
    for inference_dir in inference_dirs:
        if not inference_dir.exists():
            raise FileNotFoundError(f"Inference directory does not exist: {inference_dir}")
        if not inference_dir.is_dir():
            raise NotADirectoryError(f"Inference path is not a directory: {inference_dir}")
        files.extend(path for path in sorted(inference_dir.glob(pattern)) if path.is_file())
    return sorted(files)


def format_display_path(pred_path: Path) -> str:
    parts = pred_path.parts
    try:
        inference_index = parts.index("inference")
    except ValueError:
        return str(pred_path)
    shortened_parts = parts[inference_index + 1 :]
    return "/".join(shortened_parts)


def main() -> None:
    args = build_parser().parse_args()
    inference_dirs = tuple(args.inference_dirs) if args.inference_dirs else DEFAULT_INFERENCE_DIRS
    gold_rows = load_jsonl(args.gold_path)
    prediction_files = iter_prediction_files(inference_dirs, args.glob)

    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files matched {args.glob!r} in: "
            + ", ".join(str(path) for path in inference_dirs)
        )

    path_width = max(len("file"), *(len(format_display_path(path)) for path in prediction_files))
    header = (
        f"{'file':<{path_width}}  "
        f"{'score':>8}  "
        f"{'abs_f1':>8}  "
        f"{'full_f1':>8}  "
        f"{'missing':>7}  "
        f"{'extra':>5}"
    )
    print(header)
    print("-" * len(header))
    for pred_path in prediction_files:
        try:
            pred_rows = load_jsonl(pred_path)
            metrics = evaluate_track1(
                gold_rows=gold_rows,
                pred_rows=pred_rows,
                full_text_prefix=args.full_text_prefix,
                strict_ids=args.strict_ids,
            )
            print(
                f"{format_display_path(pred_path):<{path_width}}  "
                f"{metrics['score']:.6f}  "
                f"{metrics['abstracts_macro_f1']:.6f}  "
                f"{metrics['full_texts_macro_f1']:.6f}  "
                f"{len(metrics['missing_prediction_ids']):>7}  "
                f"{len(metrics['extra_prediction_ids']):>5}"
            )
        except Exception as error:
            payload = {
                "file": str(pred_path),
                "error": str(error),
            }
            print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
