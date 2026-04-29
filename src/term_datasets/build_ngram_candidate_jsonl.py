from __future__ import annotations

import argparse
import json
from pathlib import Path

from term_datasets.text_processing import TextProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed"
DEFAULT_SOURCE_FILENAMES = (
    "train_t1_candidates.jsonl",
    "test1_t1_candidates.jsonl",
    "test1_t3_candidates.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CL-RuTerm3 jsonl files with full n-gram candidate spans in the `candidates` field."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing source processed jsonl files.",
    )
    parser.add_argument(
        "--max-words-per-ngram",
        type=int,
        default=7,
        help="Maximum number of words in generated n-grams.",
    )
    return parser


def materialize_candidates(
    input_path: Path,
    output_path: Path,
    processor: TextProcessor,
    max_words_per_ngram: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            candidates = [
                [span.start, span.end]
                for sentence in processor.split_sentences(row["text"])
                for span in processor.extract_ngrams(sentence, max_n=max_words_per_ngram)
            ]
            row["candidates"] = candidates
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    processed_dir = args.processed_dir.expanduser().resolve()
    processor = TextProcessor()

    suffix = f"ng{args.max_words_per_ngram}_all_candidates"
    for source_name in DEFAULT_SOURCE_FILENAMES:
        input_path = processed_dir / source_name
        target_name = source_name.replace("_candidates.jsonl", f"_{suffix}.jsonl")
        output_path = processed_dir / target_name
        if not input_path.exists():
            raise FileNotFoundError(f"Source file not found: {input_path}")
        materialize_candidates(
            input_path=input_path,
            output_path=output_path,
            processor=processor,
            max_words_per_ngram=args.max_words_per_ngram,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
