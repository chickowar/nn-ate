from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from safetensors.torch import load_file as load_safetensors_file

from term_datasets.CL_RuTerm3 import CLRUTERM3_TEST1_PATH, build_span_dataset_elements
from term_datasets.CL_RuTerm3_getters import get_raw_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINDER_ROOT = PROJECT_ROOT / "external" / "fulstock-binder"
if str(BINDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BINDER_ROOT))

from src.config import BinderConfig  # type: ignore  # noqa: E402
from src.model import Binder  # type: ignore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained BINDER model and export RuTermEval jsonl predictions."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained BINDER checkpoint directory.")
    parser.add_argument(
        "--binder-config-path",
        type=Path,
        default=BINDER_ROOT / "conf" / "NN_ATE" / "track1_candidates_train.json",
        help="Path to BINDER training config JSON used to recover tokenizer/model/data settings.",
    )
    parser.add_argument("--dataset-path", type=Path, default=Path(CLRUTERM3_TEST1_PATH))
    parser.add_argument("--output-path", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--has-preprocessed-candidates",
        dest="has_preprocessed_candidates",
        action="store_true",
        help="Use candidates from JSONL candidates column.",
    )
    candidate_group.add_argument(
        "--no-preprocessed-candidates",
        dest="has_preprocessed_candidates",
        action="store_false",
        help="Ignore candidates column and build candidates locally.",
    )
    parser.set_defaults(has_preprocessed_candidates=None)
    parser.add_argument("--max-words-per-ngram", type=int, default=7)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override text max_seq_length for inference. Defaults to the value from binder config.",
    )
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=None,
        help="Override doc_stride for inference. Defaults to the value from binder config.",
    )
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


def resolve_dir_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved.resolve()


def load_binder_checkpoint(model_path: Path, device: torch.device) -> Binder:
    config = BinderConfig.from_pretrained(str(model_path))
    model = Binder(config)

    safetensors_path = model_path / "model.safetensors"
    pytorch_bin_path = model_path / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path), device="cpu")
    elif pytorch_bin_path.exists():
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys when loading BINDER checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading BINDER checkpoint: {unexpected_keys}")

    model.to(device)
    model.eval()
    return model


def load_binder_json_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_config_relative_path(config_path: Path, value: str | None) -> str | None:
    if not value:
        return value
    value_path = Path(value)
    if value_path.is_absolute():
        return str(value_path)
    if "/" in value and not any(prefix in value for prefix in ("./", ".\\", "../", "..\\")):
        return value
    return str((config_path.parent / value_path).resolve())


def load_entity_type_knowledge(
    entity_type_file: Path,
    dataset_name: str,
    dataset_entity_types: list[str],
    entity_type_key_field: str,
    entity_type_desc_field: str,
) -> tuple[list[str], list[str]]:
    entity_type_ids: list[str] = []
    entity_type_descs: list[str] = []
    with entity_type_file.open("r", encoding="utf-8") as file:
        raw_content = file.read().strip()

    if not raw_content:
        raise ValueError(f"Entity type file is empty: {entity_type_file}")

    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            records = [parsed]
        elif isinstance(parsed, list):
            records = parsed
        else:
            raise ValueError(
                f"Unsupported JSON payload in {entity_type_file}: expected object or list, got {type(parsed).__name__}"
            )
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(raw_content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Failed to parse entity type record at {entity_type_file}:{line_number}: {error}"
                ) from error

    for record in records:
        if record["dataset"] != dataset_name:
            continue
        if dataset_entity_types and record[entity_type_key_field] not in dataset_entity_types:
            continue
        entity_type_ids.append(record[entity_type_key_field])
        entity_type_descs.append(record[entity_type_desc_field])
    if not entity_type_ids:
        raise ValueError(f"No entity types found for dataset={dataset_name} in {entity_type_file}")
    return entity_type_ids, entity_type_descs


def extract_candidates(
    row: dict[str, Any],
    has_preprocessed_candidates: bool,
    max_words_per_ngram: int,
) -> list[tuple[int, int]]:
    if has_preprocessed_candidates:
        return [tuple(span) for span in row.get("candidates", [])]

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    generated = build_span_dataset_elements(
        text=row["text"],
        labels=row.get("label", []),
        sample_id=row["id"],
        preprocessed_candidates=None,
        max_words_per_ngram=max_words_per_ngram,
        negative_ratio=None,
    )
    for example in generated:
        span = (int(example["span_start"]), int(example["span_end"]))
        if span in seen:
            continue
        seen.add(span)
        candidates.append(span)
    return candidates


def find_text_bounds(sequence_ids: list[int | None], input_ids: list[int]) -> tuple[int, int]:
    text_start_index = 0
    while sequence_ids[text_start_index] != 0:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] != 0:
        text_end_index -= 1

    return text_start_index, text_end_index


def char_span_to_token_span(
    offsets: list[tuple[int, int] | None],
    text_start_index: int,
    text_end_index: int,
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    start_offset = offsets[text_start_index]
    end_offset = offsets[text_end_index]
    if start_offset is None or end_offset is None:
        return None
    if start_offset[0] > start_char or end_offset[1] < end_char:
        return None

    start_token_index, end_token_index = text_start_index, text_end_index
    while start_token_index <= text_end_index and offsets[start_token_index] is not None and offsets[start_token_index][0] <= start_char:
        start_token_index += 1
    start_token_index -= 1

    while offsets[end_token_index] is not None and offsets[end_token_index][1] >= end_char:
        end_token_index -= 1
    end_token_index += 1

    return start_token_index, end_token_index


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class BinderPredictionCollator:
    def __init__(self, type_inputs: dict[str, torch.Tensor]) -> None:
        self.type_inputs = type_inputs

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        def _pad(seq: list[int], pad_token: int = 0) -> list[int]:
            if len(seq) >= max_len:
                return seq[:max_len]
            return seq + [pad_token] * (max_len - len(seq))

        batch_size = len(features)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor([_pad(feature["input_ids"]) for feature in features], dtype=torch.long),
            "attention_mask": torch.tensor([_pad(feature["attention_mask"]) for feature in features], dtype=torch.bool),
            "example_id": [feature["example_id"] for feature in features],
            "candidate_mappings": [feature["candidate_mappings"] for feature in features],
        }
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor(
                [_pad(feature["token_type_ids"]) for feature in features],
                dtype=torch.long,
            )

        batch["type_input_ids"] = self.type_inputs["input_ids"].unsqueeze(0).repeat(batch_size, 1, 1)
        batch["type_attention_mask"] = self.type_inputs["attention_mask"].unsqueeze(0).repeat(batch_size, 1, 1)
        if "token_type_ids" in self.type_inputs:
            batch["type_token_type_ids"] = self.type_inputs["token_type_ids"].unsqueeze(0).repeat(batch_size, 1, 1)
        return batch


def prepare_features(
    raw_dataset: Any,
    tokenizer: Any,
    has_preprocessed_candidates: bool,
    max_words_per_ngram: int,
    max_seq_length: int,
    doc_stride: int,
    max_span_length: int,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    features: list[dict[str, Any]] = []
    candidates_by_id: dict[str, list[tuple[int, int]]] = {}

    for row in raw_dataset:
        candidates = extract_candidates(
            row=row,
            has_preprocessed_candidates=has_preprocessed_candidates,
            max_words_per_ngram=max_words_per_ngram,
        )
        candidates_by_id[row["id"]] = candidates

        tokenized = tokenizer(
            row["text"],
            truncation=True,
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        for feature_index in range(len(tokenized["input_ids"])):
            input_ids = tokenized["input_ids"][feature_index]
            attention_mask = tokenized["attention_mask"][feature_index]
            offsets = tokenized["offset_mapping"][feature_index]
            sequence_ids = tokenized.sequence_ids(feature_index)
            text_start_index, text_end_index = find_text_bounds(sequence_ids, input_ids)

            candidate_mappings: list[dict[str, int]] = []
            for candidate_index, (start_char, end_char) in enumerate(candidates):
                token_span = char_span_to_token_span(
                    offsets=offsets,
                    text_start_index=text_start_index,
                    text_end_index=text_end_index,
                    start_char=start_char,
                    end_char=end_char,
                )
                if token_span is None:
                    continue
                start_token_index, end_token_index = token_span
                if end_token_index - start_token_index + 1 > max_span_length:
                    continue
                candidate_mappings.append(
                    {
                        "candidate_index": candidate_index,
                        "start_token_index": start_token_index,
                        "end_token_index": end_token_index,
                    }
                )

            if not candidate_mappings:
                continue

            feature: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "example_id": row["id"],
                "candidate_mappings": candidate_mappings,
            }
            if "token_type_ids" in tokenized:
                feature["token_type_ids"] = tokenized["token_type_ids"][feature_index]
            features.append(feature)

    return features, candidates_by_id


def write_predictions(
    output_path: Path,
    raw_dataset: Any,
    candidates_by_id: dict[str, list[tuple[int, int]]],
    candidate_scores_by_id: dict[str, dict[int, float]],
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in raw_dataset:
            candidates = candidates_by_id.get(row["id"], [])
            score_map = candidate_scores_by_id.get(row["id"], {})
            candidate_probabilities = [float(score_map.get(index, 0.0)) for index in range(len(candidates))]
            positive_spans = [
                [start, end]
                for (start, end), probability in zip(candidates, candidate_probabilities)
                if probability >= threshold
            ]
            payload = {
                "id": row["id"],
                "text": row["text"],
                "label": positive_spans,
                "candidates": [[start, end] for start, end in candidates],
                "candidate_probabilities": candidate_probabilities,
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model_path = resolve_dir_path(args.model_path)
    binder_config_path = resolve_dir_path(args.binder_config_path)

    binder_run_config = load_binder_json_config(binder_config_path)
    model_name_or_path = resolve_config_relative_path(binder_config_path, binder_run_config.get("model_name_or_path"))
    entity_type_file = resolve_config_relative_path(binder_config_path, binder_run_config.get("entity_type_file"))
    if entity_type_file is None:
        raise ValueError("entity_type_file must be provided in binder config")

    raw_dataset = get_raw_dataset(args.dataset_path, flat=False)
    if args.has_preprocessed_candidates is None:
        has_preprocessed_candidates = "candidates" in raw_dataset.column_names
    else:
        has_preprocessed_candidates = args.has_preprocessed_candidates

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

    model = load_binder_checkpoint(model_path, device)
    requested_max_seq_length = args.max_seq_length or binder_run_config.get("max_seq_length", 192)
    effective_max_seq_length = min(requested_max_seq_length, tokenizer.model_max_length)
    effective_doc_stride = args.doc_stride or binder_run_config.get("doc_stride", 16)
    if effective_max_seq_length != requested_max_seq_length:
        print(
            f"Requested max_seq_length={requested_max_seq_length} exceeds tokenizer limit "
            f"{tokenizer.model_max_length}; using {effective_max_seq_length} instead."
        )

    entity_type_ids, entity_type_descs = load_entity_type_knowledge(
        entity_type_file=Path(entity_type_file),
        dataset_name=binder_run_config["dataset_name"],
        dataset_entity_types=binder_run_config.get("dataset_entity_types", []),
        entity_type_key_field=binder_run_config.get("entity_type_key_field", "name"),
        entity_type_desc_field=binder_run_config.get("entity_type_desc_field", "description"),
    )
    tokenized_descriptions = tokenizer(
        entity_type_descs,
        truncation=True,
        max_length=binder_run_config.get("max_seq_length", 192),
        padding="longest",
    )
    type_inputs = {
        "input_ids": torch.tensor(tokenized_descriptions["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized_descriptions["attention_mask"], dtype=torch.long),
    }
    if "token_type_ids" in tokenized_descriptions:
        type_inputs["token_type_ids"] = torch.tensor(tokenized_descriptions["token_type_ids"], dtype=torch.long)

    features, candidates_by_id = prepare_features(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        has_preprocessed_candidates=has_preprocessed_candidates,
        max_words_per_ngram=args.max_words_per_ngram,
        max_seq_length=effective_max_seq_length,
        doc_stride=effective_doc_stride,
        max_span_length=binder_run_config.get("max_span_length", 30),
    )

    dataloader = DataLoader(
        features,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=BinderPredictionCollator(type_inputs),
    )

    candidate_scores_by_id: dict[str, dict[int, float]] = defaultdict(dict)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting spans with BINDER"):
            example_ids = batch.pop("example_id")
            candidate_mappings = batch.pop("candidate_mappings")
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            span_scores = outputs.span_scores.detach().cpu()
            num_types = span_scores.shape[1]

            for feature_index, (example_id, mappings) in enumerate(zip(example_ids, candidate_mappings)):
                for mapping in mappings:
                    candidate_index = mapping["candidate_index"]
                    start_token_index = mapping["start_token_index"]
                    end_token_index = mapping["end_token_index"]
                    best_probability = 0.0
                    for type_index in range(num_types):
                        threshold_score = float(span_scores[feature_index, type_index, 0, 0])
                        candidate_score = float(span_scores[feature_index, type_index, start_token_index, end_token_index])
                        probability = sigmoid(candidate_score - threshold_score)
                        best_probability = max(best_probability, probability)

                    previous = candidate_scores_by_id[example_id].get(candidate_index, 0.0)
                    if best_probability > previous:
                        candidate_scores_by_id[example_id][candidate_index] = best_probability

    write_predictions(
        output_path=args.output_path,
        raw_dataset=raw_dataset,
        candidates_by_id=candidates_by_id,
        candidate_scores_by_id=candidate_scores_by_id,
        threshold=args.threshold,
    )
    print(f"Saved predictions to {args.output_path}")
    print(
        "Candidate source:",
        "preprocessed candidates/candidates" if has_preprocessed_candidates else "local n-gram extraction",
    )
    print("Inference max_seq_length:", effective_max_seq_length)
    print("Inference doc_stride:", effective_doc_stride)
    print("Entity types used:", entity_type_ids)


if __name__ == "__main__":
    main()
