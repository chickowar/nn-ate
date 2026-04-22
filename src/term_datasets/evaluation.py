from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two JSONL span datasets with exact-match F1."
    )
    parser.add_argument("--gold-path", type=Path, required=True)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument("--strict-ids", action="store_true")
    return parser


def normalize_spans(raw_spans: Any) -> set[tuple[int, int]]:
    if raw_spans is None:
        return set()

    normalized: set[tuple[int, int]] = set()
    for raw_span in raw_spans:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            raise ValueError(f"Span must be [start, end], got: {raw_span!r}")
        start, end = int(raw_span[0]), int(raw_span[1])
        if start >= end:
            raise ValueError(f"Span start must be < end, got: {(start, end)!r}")
        normalized.add((start, end))
    return normalized


def load_jsonl_spans(path: Path) -> dict[str, set[tuple[int, int]]]:
    rows: dict[str, set[tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = payload["id"]
            if doc_id in rows:
                raise ValueError(f"Duplicate id {doc_id!r} in {path} at line {line_number}")
            rows[doc_id] = normalize_spans(payload.get("label", []))
    return rows


def compute_exact_match_f1(
    gold_rows: dict[str, set[tuple[int, int]]],
    pred_rows: dict[str, set[tuple[int, int]]],
    strict_ids: bool = False,
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

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for doc_id, gold_spans in gold_rows.items():
        pred_spans = pred_rows.get(doc_id, set())
        true_positive += len(gold_spans & pred_spans)
        false_positive += len(pred_spans - gold_spans)
        false_negative += len(gold_spans - pred_spans)

    for doc_id in extra_ids:
        false_positive += len(pred_rows[doc_id])

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "gold_document_count": len(gold_rows),
        "pred_document_count": len(pred_rows),
        "missing_prediction_ids": missing_ids,
        "extra_prediction_ids": extra_ids,
    }


def main() -> None:
    args = build_parser().parse_args()
    gold_rows = load_jsonl_spans(args.gold_path)
    pred_rows = load_jsonl_spans(args.pred_path)
    metrics = compute_exact_match_f1(
        gold_rows=gold_rows,
        pred_rows=pred_rows,
        strict_ids=args.strict_ids,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
