from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
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
    parser.add_argument("--output-root", type=str, default=str(Path(__file__).parent / "outputs" / "span_classification"))
    parser.add_argument(
        "--tensorboard-root",
        type=str,
        default=str(Path(__file__).parent / "runs" / "tensorboard" / "span_classification"),
    )
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
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
    train_ratio = "all" if args.train_negative_ratio is None else str(args.train_negative_ratio)
    eval_ratio = "all" if args.eval_negative_ratio is None else str(args.eval_negative_ratio)
    model_slug = args.model_name.replace("/", "_")
    return (
        f"{model_slug}"
        f"_ng{args.max_words_per_ngram}"
        f"_len{args.max_length}"
        f"_trneg{train_ratio}"
        f"_evneg{eval_ratio}"
        f"_lr{args.learning_rate}"
        f"_bs{args.per_device_train_batch_size}"
        f"_ga{args.gradient_accumulation_steps}"
        f"_ep{args.num_train_epochs}"
        f"_seed{args.seed}"
    )


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
) -> None:
    train_ratio = train_dataset.num_rows / max(1, train_dataset.num_rows + eval_dataset.num_rows)
    eval_ratio = eval_dataset.num_rows / max(1, train_dataset.num_rows + eval_dataset.num_rows)
    print("=" * 80)
    print("Span classification training run")
    print(f"run_name: {run_name}")
    print(f"dataset_path: {args.dataset_path}")
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {output_dir}")
    print(f"tensorboard_dir: {logging_dir}")
    print(f"seed: {args.seed}")
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
) -> tuple[Dataset, Dataset, dict[str, dict[str, int]]]:
    raw_dataset: DatasetDict = get_raw_dataset(args.dataset_path, flat=False).train_test_split(
        test_size=args.test_size,
        seed=args.seed,
    )

    train_span_dataset = get_span_dataset(
        raw_dataset=raw_dataset["train"],
        max_words_per_ngram=args.max_words_per_ngram,
        negative_ratio=args.train_negative_ratio,
        seed=args.seed,
    )
    eval_span_dataset = get_span_dataset(
        raw_dataset=raw_dataset["test"],
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
    )


def compute_eval_steps(train_dataset_size: int, args: argparse.Namespace) -> int:
    effective_batch_size = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_dataset_size / effective_batch_size))
    return max(1, steps_per_epoch // max(1, args.evals_per_epoch))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    use_tf32 = not args.disable_tf32
    use_fp16 = torch.cuda.is_available() and not args.disable_fp16
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    enable_fast_math(use_tf32=use_tf32)

    run_name = make_run_name(args)
    output_dir = Path(args.output_root) / run_name
    logging_dir = Path(args.tensorboard_root) / run_name

    def get_tokenizer(model_name) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name)

    tokenizer = get_tokenizer(args.model_name)
    train_dataset, eval_dataset, dataset_summaries = build_tokenized_datasets(args, tokenizer)

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
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=True,
        fp16=use_fp16,
        tf32=use_tf32 if torch.cuda.is_available() else False,
        report_to="tensorboard",
    )

    trainer = Trainer(
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
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    save_run_config(output_dir / "eval_metrics.json", eval_metrics)
    print("Final eval metrics:", eval_metrics)


if __name__ == "__main__":
    main()
