from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from common.rutermeval_score import (
    DocumentPrediction,
    compute_f1,
    evaluate_track1,
    load_jsonl,
    normalize_spans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed"
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / "data" / "CL-RuTerm3" / "inference"


@dataclass(frozen=True)
class DatasetPair:
    display_name: str
    gold_path: Path
    prediction_path: Path
    metric_name: str


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    train_score: float
    test1_t1_score: float
    test1_t3_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep thresholds for one inference model and compare scores on "
            "train_t1, test1_t1 and test1_t3."
        )
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        required=True,
        help=(
            "Common suffix of inference files after dataset prefix, for example "
            "'BINDER_Kaggle_ruroberta_large_msl192_lr3e-05_ga8_seed38_cp3600'."
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="Explicit threshold list. If omitted, thresholds are generated uniformly.",
    )
    parser.add_argument("--threshold-count", type=int, default=10)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save detailed sweep results as JSON.",
    )
    return parser


def resolve_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        thresholds = [float(value) for value in args.thresholds]
    else:
        if args.threshold_count < 2:
            raise ValueError("--threshold-count must be at least 2 when --thresholds is not provided.")
        if args.threshold_min > args.threshold_max:
            raise ValueError("--threshold-min must be <= --threshold-max.")
        step = (args.threshold_max - args.threshold_min) / (args.threshold_count - 1)
        thresholds = [args.threshold_min + index * step for index in range(args.threshold_count)]

    normalized = []
    seen: set[float] = set()
    for value in thresholds:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}.")
        rounded = round(value, 6)
        if rounded not in seen:
            seen.add(rounded)
            normalized.append(rounded)
    return sorted(normalized)


def build_dataset_pairs(processed_dir: Path, inference_dir: Path, model_suffix: str) -> list[DatasetPair]:
    pairs = [
        DatasetPair(
            display_name="train_t1",
            gold_path=processed_dir / "train_t1_candidates.jsonl",
            prediction_path=inference_dir / f"train_t1_candidates_{model_suffix}.jsonl",
            metric_name="doc_macro_f1",
        ),
        DatasetPair(
            display_name="test1_t1",
            gold_path=processed_dir / "test1_t1_candidates.jsonl",
            prediction_path=inference_dir / f"test_t1_candidates_{model_suffix}.jsonl",
            metric_name="rutermeval_track1",
        ),
        DatasetPair(
            display_name="test1_t3",
            gold_path=processed_dir / "test1_t3_candidates.jsonl",
            prediction_path=inference_dir / f"test_t3_candidates_{model_suffix}.jsonl",
            metric_name="doc_macro_f1",
        ),
    ]

    missing_paths = [
        path
        for pair in pairs
        for path in (pair.gold_path, pair.prediction_path)
        if not path.exists()
    ]
    if missing_paths:
        joined = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required files were not found: {joined}")

    return pairs


def normalize_candidates(raw_candidates: Any) -> list[tuple[int, int]]:
    if raw_candidates is None:
        return []
    normalized: list[tuple[int, int]] = []
    for raw_span in raw_candidates:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            raise ValueError(f"Candidate span must be [start, end], got: {raw_span!r}")
        start, end = int(raw_span[0]), int(raw_span[1])
        if start >= end:
            raise ValueError(f"Candidate span start must be < end, got: {(start, end)!r}")
        normalized.append((start, end))
    return normalized


def normalize_probabilities(raw_probabilities: Any, expected_count: int) -> list[float]:
    if raw_probabilities is None:
        return [0.0] * expected_count
    probabilities = [float(value) for value in raw_probabilities]
    if len(probabilities) != expected_count:
        raise ValueError(
            "candidate_probabilities length does not match candidates length: "
            f"{len(probabilities)} != {expected_count}"
        )
    return probabilities


def build_prediction_rows(prediction_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with prediction_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = payload["id"]
            if doc_id in rows:
                raise ValueError(f"Duplicate id {doc_id!r} in {prediction_path} at line {line_number}")
            candidates = normalize_candidates(payload.get("candidates"))
            rows[doc_id] = {
                "id": doc_id,
                "text": payload.get("text"),
                "candidates": candidates,
                "candidate_probabilities": normalize_probabilities(
                    payload.get("candidate_probabilities"),
                    expected_count=len(candidates),
                ),
            }
    return rows


def select_spans(candidates: list[tuple[int, int]], probabilities: list[float], threshold: float) -> list[list[int]]:
    return [
        [start, end]
        for (start, end), probability in zip(candidates, probabilities)
        if probability >= threshold
    ]


def materialize_predictions(
    prediction_rows: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    materialized: dict[str, dict[str, Any]] = {}
    for doc_id, row in prediction_rows.items():
        materialized[doc_id] = {
            "id": doc_id,
            "text": row["text"],
            "label": select_spans(
                candidates=row["candidates"],
                probabilities=row["candidate_probabilities"],
                threshold=threshold,
            ),
        }
    return materialized


def evaluate_macro_document_f1_from_rows(
    gold_rows: dict[str, Any],
    pred_rows: dict[str, Any],
) -> dict[str, Any]:
    gold_ids = set(gold_rows)
    pred_ids = set(pred_rows)
    missing_ids = sorted(gold_ids - pred_ids)
    extra_ids = sorted(pred_ids - gold_ids)

    scores: list[float] = []
    for doc_id, gold_row in gold_rows.items():
        pred_row = pred_rows.get(doc_id)
        pred_spans = set(tuple(span) for span in pred_row.spans) if pred_row else set()
        scores.append(compute_f1(set(gold_row.spans), pred_spans))

    score = mean(scores) if scores else 0.0
    return {
        "score": score,
        "doc_macro_f1": score,
        "doc_count": len(scores),
        "missing_prediction_ids": missing_ids,
        "extra_prediction_ids": extra_ids,
    }


def score_dataset_pair(
    pair: DatasetPair,
    gold_rows: dict[str, Any],
    prediction_rows: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    threshold_rows = materialize_predictions(prediction_rows, threshold)
    pred_documents = {
        row_id: DocumentPrediction(
            doc_id=row_id,
            text=payload.get("text"),
            spans=normalize_spans(payload.get("label", [])),
        )
        for row_id, payload in threshold_rows.items()
    }

    if pair.metric_name == "rutermeval_track1":
        return evaluate_track1(
            gold_rows=gold_rows,
            pred_rows=pred_documents,
            full_text_prefix="ft-",
            strict_ids=False,
        )
    if pair.metric_name == "doc_macro_f1":
        return evaluate_macro_document_f1_from_rows(gold_rows=gold_rows, pred_rows=pred_documents)
    raise ValueError(f"Unsupported metric: {pair.metric_name}")


def run_sweep(pairs: list[DatasetPair], thresholds: list[float]) -> list[ThresholdResult]:
    gold_by_name = {pair.display_name: load_jsonl(pair.gold_path) for pair in pairs}
    predictions_by_name = {
        pair.display_name: build_prediction_rows(pair.prediction_path)
        for pair in pairs
    }

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        scores = {
            pair.display_name: score_dataset_pair(
                pair=pair,
                gold_rows=gold_by_name[pair.display_name],
                prediction_rows=predictions_by_name[pair.display_name],
                threshold=threshold,
            )["score"]
            for pair in pairs
        }
        results.append(
            ThresholdResult(
                threshold=threshold,
                train_score=scores["train_t1"],
                test1_t1_score=scores["test1_t1"],
                test1_t3_score=scores["test1_t3"],
            )
        )
    return results


def format_results_table(results: list[ThresholdResult]) -> str:
    header = ["threshold", "train_t1", "test1_t1", "test1_t3"]
    rows = [
        [
            f"{result.threshold:.4f}",
            f"{result.train_score:.6f}",
            f"{result.test1_t1_score:.6f}",
            f"{result.test1_t3_score:.6f}",
        ]
        for result in results
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]

    def format_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    return "\n".join([format_row(header), format_row(["-" * width for width in widths]), *[format_row(row) for row in rows]])


def results_to_payload(
    model_suffix: str,
    pairs: list[DatasetPair],
    results: list[ThresholdResult],
) -> dict[str, Any]:
    return {
        "model_suffix": model_suffix,
        "datasets": [
            {
                "display_name": pair.display_name,
                "gold_path": str(pair.gold_path),
                "prediction_path": str(pair.prediction_path),
                "metric_name": pair.metric_name,
            }
            for pair in pairs
        ],
        "results": [
            {
                "threshold": result.threshold,
                "train_t1": result.train_score,
                "test1_t1": result.test1_t1_score,
                "test1_t3": result.test1_t3_score,
            }
            for result in results
        ],
    }


def main() -> None:
    args = build_parser().parse_args()
    thresholds = resolve_thresholds(args)
    pairs = build_dataset_pairs(
        processed_dir=args.processed_dir,
        inference_dir=args.inference_dir,
        model_suffix=args.model_suffix,
    )
    results = run_sweep(pairs=pairs, thresholds=thresholds)
    print(format_results_table(results))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = results_to_payload(
            model_suffix=args.model_suffix,
            pairs=pairs,
            results=results,
        )
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
