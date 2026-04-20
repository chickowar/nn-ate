from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class DocumentPrediction:
    doc_id: str
    text: str | None
    spans: tuple[tuple[int, int], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute RuTermEval-2024 Track 1 score for exact term span matches."
    )
    parser.add_argument("--gold-path", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument("--full-text-prefix", type=str, default="ft-")
    parser.add_argument("--strict-ids", action="store_true")
    return parser


def normalize_spans(raw_spans: Any) -> tuple[tuple[int, int], ...]:
    if raw_spans is None:
        return ()

    normalized: set[tuple[int, int]] = set()
    for raw_span in raw_spans:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            raise ValueError(f"Span must be [start, end], got: {raw_span!r}")
        start, end = int(raw_span[0]), int(raw_span[1])
        if start >= end:
            raise ValueError(f"Span start must be < end, got: {(start, end)!r}")
        normalized.add((start, end))
    return tuple(sorted(normalized))


def load_jsonl(path: Path) -> dict[str, DocumentPrediction]:
    rows: dict[str, DocumentPrediction] = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = payload["id"]
            if doc_id in rows:
                raise ValueError(f"Duplicate id {doc_id!r} in {path} at line {line_number}")
            rows[doc_id] = DocumentPrediction(
                doc_id=doc_id,
                text=payload.get("text"),
                spans=normalize_spans(payload.get("label", [])),
            )
    return rows


def compute_f1(gold_spans: set[tuple[int, int]], pred_spans: set[tuple[int, int]]) -> float:
    true_positive = len(gold_spans & pred_spans)
    false_positive = len(pred_spans - gold_spans)
    false_negative = len(gold_spans - pred_spans)

    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        return 1.0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def evaluate_track1(
    gold_rows: dict[str, DocumentPrediction],
    pred_rows: dict[str, DocumentPrediction],
    full_text_prefix: str,
    strict_ids: bool,
) -> dict[str, Any]:
    gold_ids = set(gold_rows)
    pred_ids = set(pred_rows)

    missing_ids = sorted(gold_ids - pred_ids)
    extra_ids = sorted(pred_ids - gold_ids)
    if strict_ids and (missing_ids or extra_ids):
        raise ValueError(
            "Prediction ids do not match gold ids. "
            f"Missing: {missing_ids[:5]!r}, extra: {extra_ids[:5]!r}"
        )

    abstract_scores: list[float] = []
    full_text_scores: list[float] = []

    for doc_id, gold_row in gold_rows.items():
        pred_row = pred_rows.get(doc_id)
        pred_spans = set(pred_row.spans) if pred_row else set()
        score = compute_f1(set(gold_row.spans), pred_spans)
        if doc_id.startswith(full_text_prefix):
            full_text_scores.append(score)
        else:
            abstract_scores.append(score)

    if not abstract_scores:
        raise ValueError("No abstract documents found in gold set.")
    if not full_text_scores:
        raise ValueError("No full-text documents found in gold set.")

    abstracts_macro_f1 = mean(abstract_scores)
    full_texts_macro_f1 = mean(full_text_scores)
    track1_score = 0.5 * (abstracts_macro_f1 + full_texts_macro_f1)

    return {
        "score": track1_score,
        "abstracts_macro_f1": abstracts_macro_f1,
        "full_texts_macro_f1": full_texts_macro_f1,
        "abstract_count": len(abstract_scores),
        "full_text_count": len(full_text_scores),
        "missing_prediction_ids": missing_ids,
        "extra_prediction_ids": extra_ids,
    }


def main() -> None:
    args = build_parser().parse_args()
    gold_rows = load_jsonl(args.gold_path)
    pred_rows = load_jsonl(args.pred_path)
    metrics = evaluate_track1(
        gold_rows=gold_rows,
        pred_rows=pred_rows,
        full_text_prefix=args.full_text_prefix,
        strict_ids=args.strict_ids,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
