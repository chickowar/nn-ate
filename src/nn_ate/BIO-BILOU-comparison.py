from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import evaluate

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed, TokenizersBackend,
    EarlyStoppingCallback
)

from term_datasets.CL_RuTerm3_getters import get_train_test_split_tokenized_dataset
from term_datasets.CL_RuTerm3 import (
    CLRUTERM3_TRAIN1_PATH,
    MODEL_NAME,
    tokenize_batch_BIO,
    tokenize_batch_BILOU,
    ID2BILOU,
    BILOU2ID,
    BIO2ID,
    ID2BIO,
    SPECIAL_TOKEN_LABEL_ID,
)

@dataclass(frozen=True)
class LabelingSetup:
    name: str
    label2id: dict[str, int]
    id2label: dict[int, str]


@dataclass(frozen=True)
class ExperimentConfig:
    labeling: str
    learning_rate: float
    num_train_epochs: int = 13
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 30
    save_steps: int = 30
    save_total_limit: int = 3
    seed: int = 42
    lr_scheduler_type: str = "linear"


def build_model(model_name: str, label2id: dict[str, int], id2label: dict[int, str]):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )


def build_compute_metrics(id2label: dict[int, str]):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # token accuracy по всем не-special токенам
        mask = labels != SPECIAL_TOKEN_LABEL_ID
        correct = (preds[mask] == labels[mask]).sum()
        total = mask.sum()
        token_accuracy = float(correct / total) if total > 0 else 0.0

        # entity-level метрики для seqeval
        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(preds, labels):
            pred_tags = []
            label_tags = []

            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == SPECIAL_TOKEN_LABEL_ID:
                    continue

                pred_tags.append(id2label[int(pred_id)])
                label_tags.append(id2label[int(label_id)])

            true_predictions.append(pred_tags)
            true_labels.append(label_tags)

        seqeval_result = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": float(seqeval_result["overall_precision"]),
            "recall": float(seqeval_result["overall_recall"]),
            "f1": float(seqeval_result["overall_f1"]),
            "seqeval_accuracy": float(seqeval_result["overall_accuracy"]),
            "token_accuracy": token_accuracy,
        }

    return compute_metrics


def make_experiment_name(cfg: ExperimentConfig) -> str:
    lr_str = f"{cfg.learning_rate:.0e}".replace("+0", "").replace("+", "")
    return (
        f"{cfg.labeling}"
        f"_lr-{lr_str}"
        f"_bs-{cfg.per_device_train_batch_size}"
        f"_ga-{cfg.gradient_accumulation_steps}"
        f"_ep-{cfg.num_train_epochs}"
        f"_wd-{cfg.weight_decay}"
        f"_seed-{cfg.seed}"
    )


def train_one_experiment(
    cfg: ExperimentConfig,
    setup: LabelingSetup,
    train_ds,
    eval_ds,
    tokenizer,
    data_collator,
    outputs_root: Path,
    tb_root: Path,
):
    experiment_name = make_experiment_name(cfg)

    output_dir = outputs_root / experiment_name
    logging_dir = tb_root / experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    model = build_model(
        model_name=MODEL_NAME,
        label2id=setup.label2id,
        id2label=setup.id2label,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    os.environ['TENSORBOARD_LOGGING_DIR'] = str(logging_dir.resolve())

    args = TrainingArguments(
        output_dir=str(output_dir),

        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=1.0,

        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        report_to=["tensorboard"],

        fp16=use_fp16,
        bf16=use_bf16,

        remove_unused_columns=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(setup.id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("=" * 80)
    print(f"START EXPERIMENT: {experiment_name}")
    print(f"Output dir:  {output_dir}")
    print(f"TensorBoard: {logging_dir}")
    print("=" * 80)

    trainer.train()

    # Финальная оценка лучшей модели
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    print(f"FINAL METRICS FOR {experiment_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return {
        "experiment_name": experiment_name,
        **metrics,
    }


def main():
    outputs_root = Path("outputs")
    tb_root = Path("runs") / "tensorboard"

    outputs_root.mkdir(parents=True, exist_ok=True)
    tb_root.mkdir(parents=True, exist_ok=True)

    def get_tokenizer(name) -> TokenizersBackend:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)

    ds_getter = partial(
        get_train_test_split_tokenized_dataset,
        0.15,  # test_size
        42,    # random_state
        jsonl_path=CLRUTERM3_TRAIN1_PATH,
        tokenizer=tokenizer,
    )

    train_BIO, test_BIO = ds_getter(labeling_technique=tokenize_batch_BIO)
    train_BILOU, test_BILOU = ds_getter(labeling_technique=tokenize_batch_BILOU)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    labeling_setups = {
        "BIO": (
            LabelingSetup(
                name="BIO",
                label2id=BIO2ID,
                id2label=ID2BIO,
            ),
            train_BIO,
            test_BIO,
        ),
        "BILOU": (
            LabelingSetup(
                name="BILOU",
                label2id=BILOU2ID,
                id2label=ID2BILOU,
            ),
            train_BILOU,
            test_BILOU,
        ),
    }

    learning_rates = [5e-5, 3e-5, 2e-5, 1e-5]

    all_results = []

    for labeling_name in ["BIO", "BILOU"]:
        setup, train_ds, eval_ds = labeling_setups[labeling_name]

        for lr in learning_rates:
            cfg = ExperimentConfig(
                labeling=labeling_name,
                learning_rate=lr
            )

            result = train_one_experiment(
                cfg=cfg,
                setup=setup,
                train_ds=train_ds,
                eval_ds=eval_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
                outputs_root=outputs_root,
                tb_root=tb_root,
            )
            all_results.append(result)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 80)
    for row in all_results:
        print(row)


if __name__ == "__main__":
    main()