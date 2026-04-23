from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from term_datasets.CL_RuTerm3 import CLRUTERM3_TEST1_PATH
from term_datasets.CL_RuTerm3_getters import get_raw_dataset, get_span_dataset, tokenize_span_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained span classification model and export RuTermEval jsonl predictions."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=Path(CLRUTERM3_TEST1_PATH))
    parser.add_argument("--output-path", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--has-preprocessed-candidates",
        dest="has_preprocessed_candidates",
        action="store_true",
        help="Use candidate spans from JSONL candidates/candidates column.",
    )
    candidate_group.add_argument(
        "--no-preprocessed-candidates",
        dest="has_preprocessed_candidates",
        action="store_false",
        help="Ignore candidates/candidates column and build candidates locally.",
    )
    parser.set_defaults(has_preprocessed_candidates=None)
    parser.add_argument("--max-words-per-ngram", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--tokenize-batch-size", type=int, default=1000)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_path(model_path: Path) -> str:
    resolved_path = model_path.expanduser()
    if resolved_path.exists():
        if not resolved_path.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {resolved_path}")
        return str(resolved_path.resolve())

    raise FileNotFoundError(
        "Model path does not exist: "
        f"{resolved_path}. "
        "If you mean a local checkpoint, pass the real directory path "
        "(for example, src/nn_ate/outputs/.../best)."
    )


class SpanPredictionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        metadata = {
            "id": [feature["id"] for feature in features],
            "span_start": [int(feature["span_start"]) for feature in features],
            "span_end": [int(feature["span_end"]) for feature in features],
        }
        model_features = []
        for feature in features:
            model_features.append(
                {
                    key: value
                    for key, value in feature.items()
                    if key not in {"id", "span_start", "span_end", "label"}
                }
            )
        batch = self._collator(model_features)
        batch.update(metadata)
        return batch


def write_predictions(
    output_path: Path,
    raw_dataset: Dataset,
    candidate_scores_by_id: dict[str, list[tuple[int, int, float]]],
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in raw_dataset:
            candidate_scores = candidate_scores_by_id.get(row["id"], [])
            candidates = [[start, end] for start, end, _ in candidate_scores]
            candidate_probabilities = [probability for _, _, probability in candidate_scores]
            positive_spans = [
                [start, end]
                for start, end, probability in candidate_scores
                if probability >= threshold
            ]
            payload = {
                "id": row["id"],
                "text": row["text"],
                "label": positive_spans,
                "candidates": candidates,
                "candidate_probabilities": candidate_probabilities,
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model_path = resolve_model_path(args.model_path)

    raw_dataset = get_raw_dataset(args.dataset_path, flat=False)
    if args.has_preprocessed_candidates is None:
        has_preprocessed_candidates = "candidates" in raw_dataset.column_names
    else:
        has_preprocessed_candidates = args.has_preprocessed_candidates

    span_dataset = get_span_dataset(
        raw_dataset=raw_dataset,
        has_preprocessed_candidates=has_preprocessed_candidates,
        max_words_per_ngram=args.max_words_per_ngram,
        negative_ratio=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenized_dataset = tokenize_span_dataset(
        dataset=span_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenize_batch_size,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=SpanPredictionCollator(tokenizer),
    )

    candidate_scores_by_id: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting spans"):
            metadata_ids = batch.pop("id")
            metadata_starts = batch.pop("span_start")
            metadata_ends = batch.pop("span_end")
            batch = {key: value.to(device) for key, value in batch.items()}

            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

            for doc_id, start, end, probability in zip(metadata_ids, metadata_starts, metadata_ends, probabilities):
                candidate_scores_by_id[doc_id].append((int(start), int(end), float(probability)))

    write_predictions(
        output_path=args.output_path,
        raw_dataset=raw_dataset,
        candidate_scores_by_id=candidate_scores_by_id,
        threshold=args.threshold,
    )
    print(f"Saved predictions to {args.output_path}")
    print(
        "Candidate source:",
        "preprocessed candidates/candidates" if has_preprocessed_candidates else "local n-gram extraction",
    )


if __name__ == "__main__":
    # main()
    ...
    ...