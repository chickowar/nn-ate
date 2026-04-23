from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from safetensors.torch import load_file as load_safetensors_file
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from nn_ate.bert_getters import DEFAULT_MODEL, get_bert_sequence_classification
from term_datasets.CL_RuTerm3 import CLRUTERM3_TRAIN1_PATH, SPAN_ID2LABEL, SPAN_LABEL2ID
from term_datasets.CL_RuTerm3_getters import get_raw_dataset, get_span_dataset, tokenize_span_dataset


def parse_optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a span classification model for CL-RuTerm3.")
    parser.add_argument("--dataset-path", type=str, default=CLRUTERM3_TRAIN1_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume Trainer state from a checkpoint directory (for example, .../checkpoint-2136).",
    )
    archive_group = parser.add_mutually_exclusive_group()
    archive_group.add_argument(
        "--archive-existing-checkpoints",
        dest="archive_existing_checkpoints",
        action="store_true",
        help="Rename pre-existing checkpoint-* directories to old-* before resume training.",
    )
    archive_group.add_argument(
        "--no-archive-existing-checkpoints",
        dest="archive_existing_checkpoints",
        action="store_false",
        help="Keep pre-existing checkpoint-* directories unchanged while resuming training.",
    )
    parser.set_defaults(archive_existing_checkpoints=True)
    parser.add_argument("--output-root", type=str, default=str(Path(__file__).parent / "outputs" / "span_classification"))
    parser.add_argument(
        "--tensorboard-root",
        type=str,
        default=str(Path(__file__).parent / "runs" / "tensorboard" / "span_classification"),
    )
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--train-negative-ratio", type=parse_optional_float, default=2.0)
    parser.add_argument("--eval-negative-ratio", type=parse_optional_float, default=None)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evals-per-epoch", type=int, default=1)
    parser.add_argument("--max-eval-to-train-ratio", type=float, default=None)

    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-eval-rebalance", action="store_true")
    return parser


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    model_slug = args.model_name.replace("/", "_")
    run_name = [model_slug]
    if args.has_preprocessed_candidates is True:
        run_name.append("candpre")
    elif args.has_preprocessed_candidates is False:
        train_ratio = "all" if args.train_negative_ratio is None else str(args.train_negative_ratio)
        eval_ratio = "all" if args.eval_negative_ratio is None else str(args.eval_negative_ratio)
        run_name.extend([
            "candlocal",
            f"ng{args.max_words_per_ngram}",
            f"trneg{train_ratio}",
            f"evneg{eval_ratio}",
        ])
    else:
        run_name.append("candauto")

    run_name.extend([
        f"len{args.max_length}",
        f"lr{args.learning_rate}",
        f"bs{args.per_device_train_batch_size}",
        f"ga{args.gradient_accumulation_steps}",
        f"ep{args.num_train_epochs}",
        f"seed{args.seed}",
    ])
    return "_".join(run_name)


def enable_fast_math(use_tf32: bool) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32


def rebalance_eval_dataset(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    test_size: float,
    seed: int,
    max_eval_to_train_ratio: float | None = None,
) -> Dataset:
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        return eval_dataset

    target_ratio = max_eval_to_train_ratio
    if target_ratio is None:
        target_ratio = (test_size / max(1e-8, 1.0 - test_size)) * 1.1

    max_eval_size = max(1, int(round(len(train_dataset) * target_ratio)))
    if len(eval_dataset) <= max_eval_size:
        return eval_dataset

    labels = eval_dataset["label"]
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    if len(positive_indices) >= max_eval_size or not negative_indices:
        return eval_dataset

    rng = np.random.default_rng(seed)
    kept_negative_count = min(len(negative_indices), max_eval_size - len(positive_indices))
    sampled_negative_indices = rng.choice(negative_indices, size=kept_negative_count, replace=False).tolist()
    selected_indices = sorted(positive_indices + [int(index) for index in sampled_negative_indices])
    return eval_dataset.select(selected_indices)


def prepare_trainer_dataset(dataset: Dataset) -> Dataset:
    prepared = dataset.rename_column("label", "labels")
    removable_columns = [column for column in ("id", "span_start", "span_end") if column in prepared.column_names]
    if removable_columns:
        prepared = prepared.remove_columns(removable_columns)
    return prepared


def compute_binary_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def describe_dataset(name: str, dataset: Dataset) -> dict[str, int]:
    labels = dataset["label"]
    positives = int(sum(labels))
    negatives = len(labels) - positives
    summary = {
        "size": len(dataset),
        "positives": positives,
        "negatives": negatives,
    }
    print(f"{name}: size={summary['size']}, positives={summary['positives']}, negatives={summary['negatives']}")
    return summary


def save_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_explicit_cli_destinations(parser: argparse.ArgumentParser) -> set[str]:
    explicit_options = {token.split("=", maxsplit=1)[0] for token in sys.argv[1:] if token.startswith("--")}
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    return {option_to_dest[option] for option in explicit_options if option in option_to_dest}


def resolve_resume_checkpoint(checkpoint_path: str | None) -> Path | None:
    if checkpoint_path is None:
        return None

    resolved_path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {resolved_path}")

    required_files = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing_files = [filename for filename in required_files if not (resolved_path / filename).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Checkpoint directory does not contain files required for resume training: {missing}"
        )
    return resolved_path


def load_checkpoint_training_args(checkpoint_dir: Path) -> Any | None:
    training_args_path = checkpoint_dir / "training_args.bin"
    if not training_args_path.exists():
        return None
    try:
        return torch.load(training_args_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(training_args_path, map_location="cpu")


def normalize_checkpoint_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
        if normalized_key not in normalized:
            normalized[normalized_key] = value
    return normalized


def load_checkpoint_model_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor] | None:
    safe_weights_path = checkpoint_dir / "model.safetensors"
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if safe_weights_path.exists():
        return normalize_checkpoint_state_dict_keys(load_safetensors_file(str(safe_weights_path), device="cpu"))
    if weights_path.exists():
        return normalize_checkpoint_state_dict_keys(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
    return None


def apply_resume_defaults(
    args: argparse.Namespace,
    explicit_destinations: set[str],
    resume_checkpoint: Path,
) -> dict[str, Any]:
    resume_run_dir = resume_checkpoint.parent
    run_config_path = resume_run_dir / "run_config.json"
    checkpoint_training_args = load_checkpoint_training_args(resume_checkpoint)
    run_config_args: dict[str, Any] = {}
    if run_config_path.exists():
        run_config_payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config_args = run_config_payload.get("args", {})

    def assign_if_implicit(dest: str, value: Any) -> None:
        if dest not in explicit_destinations and value is not None:
            setattr(args, dest, value)

    for dest in (
        "dataset_path",
        "test_size",
        "seed",
        "has_preprocessed_candidates",
        "max_words_per_ngram",
        "max_length",
        "tokenize_batch_size",
        "train_negative_ratio",
        "eval_negative_ratio",
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "warmup_ratio",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_total_limit",
        "evals_per_epoch",
        "max_eval_to_train_ratio",
        "disable_tf32",
        "disable_fp16",
        "no_gradient_checkpointing",
        "disable_eval_rebalance",
        "tensorboard_root",
    ):
        assign_if_implicit(dest, run_config_args.get(dest))

    if checkpoint_training_args is not None:
        checkpoint_overrides = {
            "learning_rate": getattr(checkpoint_training_args, "learning_rate", None),
            "weight_decay": getattr(checkpoint_training_args, "weight_decay", None),
            "num_train_epochs": getattr(checkpoint_training_args, "num_train_epochs", None),
            "warmup_ratio": getattr(checkpoint_training_args, "warmup_ratio", None),
            "per_device_train_batch_size": getattr(checkpoint_training_args, "per_device_train_batch_size", None),
            "per_device_eval_batch_size": getattr(checkpoint_training_args, "per_device_eval_batch_size", None),
            "gradient_accumulation_steps": getattr(checkpoint_training_args, "gradient_accumulation_steps", None),
            "logging_steps": getattr(checkpoint_training_args, "logging_steps", None),
            "save_total_limit": getattr(checkpoint_training_args, "save_total_limit", None),
        }
        for dest, value in checkpoint_overrides.items():
            assign_if_implicit(dest, value)

    if "model_name" not in explicit_destinations:
        args.model_name = str(resume_checkpoint)
    if "run_name" not in explicit_destinations and args.run_name is None:
        args.run_name = resume_run_dir.name
    if "output_root" not in explicit_destinations:
        args.output_root = str(resume_run_dir.parent)
    if "tensorboard_root" not in explicit_destinations and not run_config_args.get("tensorboard_root"):
        args.tensorboard_root = str(Path(args.tensorboard_root))
    if "save_total_limit" not in explicit_destinations:
        args.save_total_limit = None

    return {
        "resume_run_dir": resume_run_dir,
        "run_config_path": run_config_path if run_config_path.exists() else None,
        "checkpoint_training_args_path": str(resume_checkpoint / "training_args.bin")
        if (resume_checkpoint / "training_args.bin").exists()
        else None,
        "original_model_name": run_config_args.get("model_name"),
    }


def make_archived_checkpoint_name(checkpoint_dir: Path) -> Path:
    candidate = checkpoint_dir.with_name(f"old-{checkpoint_dir.name}")
    suffix = 1
    while candidate.exists():
        candidate = checkpoint_dir.with_name(f"old-{checkpoint_dir.name}-{suffix}")
        suffix += 1
    return candidate


def archive_existing_checkpoints(output_dir: Path, active_checkpoint: Path | None) -> list[tuple[Path, Path]]:
    archived: list[tuple[Path, Path]] = []
    active_checkpoint_resolved = active_checkpoint.resolve(strict=False) if active_checkpoint is not None else None
    for checkpoint_dir in sorted(output_dir.glob("checkpoint-*")):
        if not checkpoint_dir.is_dir():
            continue
        if active_checkpoint_resolved is not None and checkpoint_dir.resolve(strict=False) == active_checkpoint_resolved:
            continue
        archived_path = make_archived_checkpoint_name(checkpoint_dir)
        checkpoint_dir.rename(archived_path)
        archived.append((checkpoint_dir, archived_path))
    return archived


def print_run_summary(
    args: argparse.Namespace,
    run_name: str,
    output_dir: Path,
    logging_dir: Path,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    dataset_summaries: dict[str, dict[str, int]],
    eval_steps: int,
    use_fp16: bool,
    use_tf32: bool,
    use_gradient_checkpointing: bool,
    has_preprocessed_candidates: bool,
    resume_checkpoint: Path | None,
    archived_checkpoints: list[tuple[Path, Path]],
) -> None:
    train_ratio = train_dataset.num_rows / max(1, train_dataset.num_rows + eval_dataset.num_rows)
    eval_ratio = eval_dataset.num_rows / max(1, train_dataset.num_rows + eval_dataset.num_rows)
    print("=" * 80)
    print("Span classification training run")
    print(f"run_name: {run_name}")
    print(f"dataset_path: {args.dataset_path}")
    print(f"model_name: {args.model_name}")
    print(f"resume_from_checkpoint: {resume_checkpoint if resume_checkpoint is not None else 'None'}")
    print(f"output_dir: {output_dir}")
    print(f"tensorboard_dir: {logging_dir}")
    print(f"seed: {args.seed}")
    print(
        f"save_total_limit: {args.save_total_limit if args.save_total_limit is not None else 'disabled during resume'}"
    )
    if archived_checkpoints:
        archived_pairs = ", ".join(f"{src.name}->{dst.name}" for src, dst in archived_checkpoints)
        print(f"archived_checkpoints: {archived_pairs}")
    else:
        print("archived_checkpoints: none")
    print(
        "candidate_source: "
        f"{'preprocessed candidates/candidates' if has_preprocessed_candidates else 'local n-gram extraction'}"
    )
    print(
        f"train_size: {train_dataset.num_rows} | eval_size: {eval_dataset.num_rows} | "
        f"train_share: {train_ratio:.3f} | eval_share: {eval_ratio:.3f}"
    )
    print(
        "train_before_rebalance: "
        f"{dataset_summaries['train_before_rebalance']['size']} "
        f"(+{dataset_summaries['train_before_rebalance']['positives']} / "
        f"-{dataset_summaries['train_before_rebalance']['negatives']})"
    )
    print(
        "eval_before_rebalance: "
        f"{dataset_summaries['eval_before_rebalance']['size']} "
        f"(+{dataset_summaries['eval_before_rebalance']['positives']} / "
        f"-{dataset_summaries['eval_before_rebalance']['negatives']})"
    )
    print(
        "eval_after_rebalance: "
        f"{dataset_summaries['eval_after_rebalance']['size']} "
        f"(+{dataset_summaries['eval_after_rebalance']['positives']} / "
        f"-{dataset_summaries['eval_after_rebalance']['negatives']})"
    )
    if has_preprocessed_candidates:
        print(f"max_length: {args.max_length} | tokenize_batch_size: {args.tokenize_batch_size}")
        print("local candidate generation params: disabled")
    else:
        print(
            f"max_words_per_ngram: {args.max_words_per_ngram} | max_length: {args.max_length} | "
            f"tokenize_batch_size: {args.tokenize_batch_size}"
        )
        print(
            f"train_negative_ratio: {args.train_negative_ratio} | "
            f"eval_negative_ratio: {args.eval_negative_ratio}"
        )
    print(
        f"train_batch_size: {args.per_device_train_batch_size} | "
        f"eval_batch_size: {args.per_device_eval_batch_size} | "
        f"grad_accumulation: {args.gradient_accumulation_steps}"
    )
    print(
        f"learning_rate: {args.learning_rate} | weight_decay: {args.weight_decay} | "
        f"epochs: {args.num_train_epochs} | warmup_ratio: {args.warmup_ratio}"
    )
    print(
        f"fp16: {use_fp16} | tf32: {use_tf32} | "
        f"gradient_checkpointing: {use_gradient_checkpointing} | eval_steps: {eval_steps}"
    )
    print("=" * 80)


def build_tokenized_datasets(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset, dict[str, dict[str, int]], bool]:
    raw_dataset: DatasetDict = get_raw_dataset(args.dataset_path, flat=False).train_test_split(
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.has_preprocessed_candidates is None:
        has_preprocessed_candidates = "candidates" in raw_dataset["train"].column_names
    else:
        has_preprocessed_candidates = args.has_preprocessed_candidates

    if has_preprocessed_candidates:
        train_span_dataset = get_span_dataset(
            raw_dataset=raw_dataset["train"],
            has_preprocessed_candidates=True,
        )
        eval_span_dataset = get_span_dataset(
            raw_dataset=raw_dataset["test"],
            has_preprocessed_candidates=True,
        )
    else:
        train_span_dataset = get_span_dataset(
            raw_dataset=raw_dataset["train"],
            has_preprocessed_candidates=False,
            max_words_per_ngram=args.max_words_per_ngram,
            negative_ratio=args.train_negative_ratio,
            seed=args.seed,
        )
        eval_span_dataset = get_span_dataset(
            raw_dataset=raw_dataset["test"],
            has_preprocessed_candidates=False,
            max_words_per_ngram=args.max_words_per_ngram,
            negative_ratio=args.eval_negative_ratio,
            seed=args.seed + 1,
        )

    summaries = {
        "train_before_rebalance": describe_dataset("train_before_rebalance", train_span_dataset),
        "eval_before_rebalance": describe_dataset("eval_before_rebalance", eval_span_dataset),
    }

    if not args.disable_eval_rebalance:
        eval_span_dataset = rebalance_eval_dataset(
            train_dataset=train_span_dataset,
            eval_dataset=eval_span_dataset,
            test_size=args.test_size,
            seed=args.seed + 2,
            max_eval_to_train_ratio=args.max_eval_to_train_ratio,
        )

    summaries["eval_after_rebalance"] = describe_dataset("eval_after_rebalance", eval_span_dataset)

    tokenized_train_dataset = tokenize_span_dataset(
        dataset=train_span_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenize_batch_size,
    )
    tokenized_eval_dataset = tokenize_span_dataset(
        dataset=eval_span_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenize_batch_size,
    )

    return (
        prepare_trainer_dataset(tokenized_train_dataset),
        prepare_trainer_dataset(tokenized_eval_dataset),
        summaries,
        has_preprocessed_candidates,
    )


def compute_eval_steps(train_dataset_size: int, args: argparse.Namespace) -> int:
    effective_batch_size = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_dataset_size / effective_batch_size))
    return max(1, steps_per_epoch // max(1, args.evals_per_epoch))


class ResumeCompatibleTrainer(Trainer):
    def _load_normalized_checkpoint(self, checkpoint_dir: str, model: torch.nn.Module | None = None) -> bool:
        target_model = self.model if model is None else model
        state_dict = load_checkpoint_model_state_dict(Path(checkpoint_dir))
        if state_dict is None:
            return False
        load_result = target_model.load_state_dict(state_dict, strict=False)
        self._issue_warnings_after_load(load_result)
        return True

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: torch.nn.Module | None = None) -> None:
        if self._load_normalized_checkpoint(resume_from_checkpoint, model=model):
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_best_model(self) -> None:
        best_model_checkpoint = getattr(self.state, "best_model_checkpoint", None)
        if best_model_checkpoint and self._load_normalized_checkpoint(best_model_checkpoint):
            return
        super()._load_best_model()

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        for parameter, state in optimizer.state.items():
            for key, value in list(state.items()):
                if not torch.is_tensor(value):
                    continue
                desired_dtype = value.dtype if key == "step" else parameter.dtype
                normalized_value = value.to(device=parameter.device, dtype=desired_dtype)
                if normalized_value.layout == torch.strided:
                    normalized_value = normalized_value.contiguous()
                state[key] = normalized_value


def main() -> None:
    parser = build_parser()
    explicit_destinations = get_explicit_cli_destinations(parser)
    args = parser.parse_args()
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint)
    resume_metadata: dict[str, Any] = {}

    if resume_checkpoint is not None:
        resume_metadata = apply_resume_defaults(
            args=args,
            explicit_destinations=explicit_destinations,
            resume_checkpoint=resume_checkpoint,
        )

    set_seed(args.seed)
    use_tf32 = not args.disable_tf32
    use_fp16 = torch.cuda.is_available() and not args.disable_fp16
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    enable_fast_math(use_tf32=use_tf32)

    def get_tokenizer(model_name) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name)

    tokenizer = get_tokenizer(args.model_name)
    train_dataset, eval_dataset, dataset_summaries, has_preprocessed_candidates = build_tokenized_datasets(args, tokenizer)

    if args.run_name is None and args.has_preprocessed_candidates is None:
        args.has_preprocessed_candidates = has_preprocessed_candidates

    run_name = make_run_name(args)
    output_dir = Path(args.output_root) / run_name
    logging_dir = Path(args.tensorboard_root) / run_name
    archived_checkpoints: list[tuple[Path, Path]] = []
    if resume_checkpoint is not None and args.archive_existing_checkpoints:
        output_dir.mkdir(parents=True, exist_ok=True)
        archived_checkpoints = archive_existing_checkpoints(output_dir=output_dir, active_checkpoint=resume_checkpoint)

    model = get_bert_sequence_classification(
        model_name_or_path=args.model_name,
        id2label=SPAN_ID2LABEL,
        label2id=SPAN_LABEL2ID,
    )
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    eval_steps = compute_eval_steps(len(train_dataset), args)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        logging_dir=str(logging_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=True,
        fp16=use_fp16,
        tf32=use_tf32 if torch.cuda.is_available() else False,
        report_to="tensorboard",
    )

    trainer = ResumeCompatibleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
    )

    config_payload = {
        "run_name": run_name,
        "args": vars(args),
        "dataset_summaries": dataset_summaries,
        "final_train_size": len(train_dataset),
        "final_eval_size": len(eval_dataset),
        "eval_steps": eval_steps,
        "fp16": use_fp16,
        "tf32": use_tf32 if torch.cuda.is_available() else False,
        "gradient_checkpointing": use_gradient_checkpointing,
        "has_preprocessed_candidates": has_preprocessed_candidates,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "archived_checkpoints": [
            {"from": str(source), "to": str(target)}
            for source, target in archived_checkpoints
        ],
        "resume_metadata": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in resume_metadata.items()
        },
    }
    save_run_config(output_dir / "run_config.json", config_payload)
    print_run_summary(
        args=args,
        run_name=run_name,
        output_dir=output_dir,
        logging_dir=logging_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_summaries=dataset_summaries,
        eval_steps=eval_steps,
        use_fp16=use_fp16,
        use_tf32=use_tf32 if torch.cuda.is_available() else False,
        use_gradient_checkpointing=use_gradient_checkpointing,
        has_preprocessed_candidates=has_preprocessed_candidates,
        resume_checkpoint=resume_checkpoint,
        archived_checkpoints=archived_checkpoints,
    )

    trainer.train(resume_from_checkpoint=None if resume_checkpoint is None else str(resume_checkpoint))
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    save_run_config(output_dir / "eval_metrics.json", eval_metrics)
    print("Final eval metrics:", eval_metrics)


if __name__ == "__main__":
    main()
