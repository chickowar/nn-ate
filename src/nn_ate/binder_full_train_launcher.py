from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.rutermeval_score import compute_f1, evaluate_track1, load_jsonl

BINDER_ROOT = PROJECT_ROOT / "external" / "fulstock-binder"
DEFAULT_BASE_CONFIG_PATH = (
    BINDER_ROOT / "conf" / "NN_ATE" / "track1_candidates_fulltrain_ruroberta_large_lr3e-05_bs8_ep128.json"
)
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed" / "train_t1_candidates.jsonl"
DEFAULT_TEST_T1_FILE = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed" / "test1_t1_candidates.jsonl"
DEFAULT_TEST_T3_FILE = PROJECT_ROOT / "data" / "CL-RuTerm3" / "processed" / "test1_t3_candidates.jsonl"

EVAL_DIRNAME = "checkpoint_eval"
BEST_DIRNAME = "best_models"
RECORDS_FILENAME = "checkpoint_scores.jsonl"
GENERATED_CONFIG_FILENAME = "launcher_train_config.json"
LAUNCHER_DIRNAME = "_launcher"


@dataclass(frozen=True)
class EvaluationRecord:
    checkpoint_name: str
    checkpoint_path: str
    global_step: int | None
    source_kind: str
    test_t1: dict[str, Any]
    test_t3: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-train BINDER on train_t1_candidates, score each checkpoint on test1_t1 and test1_t3, "
            "and snapshot the best checkpoints for both metrics."
        )
    )
    parser.add_argument("--base-config-path", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-t1-file", type=Path, default=DEFAULT_TEST_T1_FILE)
    parser.add_argument("--test-t3-file", type=Path, default=DEFAULT_TEST_T3_FILE)
    parser.add_argument("--num-train-epochs", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--target-global-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--single-gpu", action="store_true", default=True)
    parser.add_argument("--allow-multi-gpu", dest="single_gpu", action="store_false")
    return parser


def resolve_device_name(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_num_devices(device_name: str, single_gpu: bool = False) -> int:
    if device_name != "cuda" or not torch.cuda.is_available():
        return 1
    if single_gpu:
        return 1
    return max(1, torch.cuda.device_count())


def build_child_env(device_name: str, single_gpu: bool) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing_pythonpath else str(SRC_ROOT) + os.pathsep + existing_pythonpath
    if device_name == "cuda" and single_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def compute_batch_settings(args: argparse.Namespace, device_name: str) -> tuple[int, int]:
    if args.per_device_train_batch_size is not None and args.gradient_accumulation_steps is not None:
        return args.per_device_train_batch_size, args.gradient_accumulation_steps

    if args.per_device_train_batch_size is not None:
        num_devices = resolve_num_devices(device_name, single_gpu=args.single_gpu)
        global_batch = args.per_device_train_batch_size * num_devices
        gradient_accumulation_steps = max(1, math.ceil(args.target_global_train_batch_size / global_batch))
        return args.per_device_train_batch_size, gradient_accumulation_steps

    num_devices = resolve_num_devices(device_name, single_gpu=args.single_gpu)
    per_device = max(1, math.ceil(args.target_global_train_batch_size / num_devices))
    effective_global = per_device * num_devices
    gradient_accumulation_steps = max(1, math.ceil(args.target_global_train_batch_size / effective_global))
    return per_device, gradient_accumulation_steps


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_output_dir(base_config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    output_dir_value = base_config.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise ValueError("Base config must contain output_dir.")
    output_dir_path = Path(output_dir_value)
    if not output_dir_path.is_absolute():
        output_dir_path = (args.base_config_path.parent / output_dir_path).resolve()
    return output_dir_path


def resolve_config_path_value(base_config_path: Path, value: str | None) -> str | None:
    if not value:
        return value
    value_path = Path(value)
    if value_path.is_absolute():
        return str(value_path)
    if "/" in value and not any(prefix in value for prefix in ("./", ".\\", "../", "..\\")):
        return value
    return str((base_config_path.parent / value_path).resolve())


def build_generated_config(base_config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    device_name = resolve_device_name(args.device)
    per_device_train_batch_size, gradient_accumulation_steps = compute_batch_settings(args, device_name)

    config = dict(base_config)
    config["train_file"] = str(args.train_file.expanduser().resolve())
    config["output_dir"] = str(output_dir)
    for path_key in [
        "entity_type_file",
        "cache_dir",
        "logging_dir",
        "binder_model_name_or_path",
        "config_name",
        "tokenizer_name",
    ]:
        if path_key in config:
            config[path_key] = resolve_config_path_value(args.base_config_path, config.get(path_key))
    config["num_train_epochs"] = args.num_train_epochs
    config["learning_rate"] = args.learning_rate
    config["per_device_train_batch_size"] = per_device_train_batch_size
    config["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    config["gradient_accumulation_steps"] = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else gradient_accumulation_steps
    )
    config["save_steps"] = args.save_steps
    config["save_total_limit"] = args.save_total_limit
    config["logging_steps"] = args.logging_steps
    config["overwrite_output_dir"] = False
    config["do_train"] = True
    config["do_eval"] = False
    config["do_predict"] = False
    config["load_best_model_at_end"] = False
    config["evaluation_strategy"] = "no"
    config.pop("validation_file", None)
    config.pop("validation_split_ratio", None)
    config.pop("validation_split_seed", None)
    config.pop("eval_steps", None)
    if args.run_name:
        config["run_name"] = args.run_name
    return config


def parse_checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    try:
        return int(path.name.split("-", 1)[1])
    except ValueError:
        return None


def is_model_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def final_checkpoint_name(output_dir: Path) -> str:
    weight_path = output_dir / "model.safetensors"
    if not weight_path.exists():
        weight_path = output_dir / "pytorch_model.bin"
    if not weight_path.exists():
        return "final"
    return f"final-{int(weight_path.stat().st_mtime)}"


def iter_candidate_model_dirs(output_dir: Path, include_final: bool) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for checkpoint_dir in sorted(output_dir.glob("checkpoint-*"), key=lambda path: parse_checkpoint_step(path) or -1):
        if is_model_dir_ready(checkpoint_dir):
            candidates.append((checkpoint_dir.name, checkpoint_dir))
    if include_final and is_model_dir_ready(output_dir):
        candidates.append((final_checkpoint_name(output_dir), output_dir))
    return candidates


def load_existing_records(records_path: Path) -> list[EvaluationRecord]:
    if not records_path.exists():
        return []
    records: list[EvaluationRecord] = []
    with records_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records.append(EvaluationRecord(**payload))
    return records


def append_record(records_path: Path, record: EvaluationRecord) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def evaluate_macro_document_f1(gold_path: Path, pred_path: Path) -> dict[str, Any]:
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)

    gold_ids = set(gold_rows)
    pred_ids = set(pred_rows)
    missing_ids = sorted(gold_ids - pred_ids)
    extra_ids = sorted(pred_ids - gold_ids)

    scores: list[float] = []
    for doc_id, gold_row in gold_rows.items():
        pred_row = pred_rows.get(doc_id)
        pred_spans = set(pred_row.spans) if pred_row else set()
        scores.append(compute_f1(set(gold_row.spans), pred_spans))

    return {
        "score": mean(scores) if scores else 0.0,
        "doc_macro_f1": mean(scores) if scores else 0.0,
        "doc_count": len(scores),
        "missing_prediction_ids": missing_ids,
        "extra_prediction_ids": extra_ids,
    }


def run_prediction(
    model_path: Path,
    binder_config_path: Path,
    dataset_path: Path,
    output_path: Path,
    threshold: float,
    device_name: str,
    single_gpu: bool,
) -> None:
    env = build_child_env(device_name=device_name, single_gpu=single_gpu)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "nn_ate" / "binder_predict.py"),
        "--model-path",
        str(model_path),
        "--binder-config-path",
        str(binder_config_path),
        "--dataset-path",
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--threshold",
        str(threshold),
        "--device",
        device_name,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def evaluate_checkpoint(
    checkpoint_name: str,
    checkpoint_path: Path,
    generated_config_path: Path,
    output_dir: Path,
    test_t1_file: Path,
    test_t3_file: Path,
    threshold: float,
    device_name: str,
    single_gpu: bool,
) -> EvaluationRecord:
    eval_dir = output_dir / EVAL_DIRNAME / checkpoint_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    t1_pred_path = eval_dir / "test1_t1_predictions.jsonl"
    t3_pred_path = eval_dir / "test1_t3_predictions.jsonl"

    run_prediction(
        model_path=checkpoint_path,
        binder_config_path=generated_config_path,
        dataset_path=test_t1_file,
        output_path=t1_pred_path,
        threshold=threshold,
        device_name=device_name,
        single_gpu=single_gpu,
    )
    run_prediction(
        model_path=checkpoint_path,
        binder_config_path=generated_config_path,
        dataset_path=test_t3_file,
        output_path=t3_pred_path,
        threshold=threshold,
        device_name=device_name,
        single_gpu=single_gpu,
    )

    t1_metrics = evaluate_track1(
        gold_rows=load_jsonl(test_t1_file),
        pred_rows=load_jsonl(t1_pred_path),
        full_text_prefix="ft-",
        strict_ids=False,
    )
    t3_metrics = evaluate_macro_document_f1(test_t3_file, t3_pred_path)

    write_json(eval_dir / "test1_t1_metrics.json", t1_metrics)
    write_json(eval_dir / "test1_t3_metrics.json", t3_metrics)

    return EvaluationRecord(
        checkpoint_name=checkpoint_name,
        checkpoint_path=str(checkpoint_path),
        global_step=parse_checkpoint_step(checkpoint_path),
        source_kind="final" if checkpoint_name.startswith("final") else "checkpoint",
        test_t1=t1_metrics,
        test_t3=t3_metrics,
    )


def copy_model_snapshot(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    if source.name.startswith("checkpoint-"):
        shutil.copytree(source, destination)
        return

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name.startswith("checkpoint-")}
        ignored.update({EVAL_DIRNAME, BEST_DIRNAME, "__pycache__"})
        return ignored

    shutil.copytree(source, destination, ignore=_ignore)


def maybe_update_best_snapshots(output_dir: Path, records: list[EvaluationRecord]) -> None:
    if not records:
        return

    best_dir = output_dir / BEST_DIRNAME
    best_dir.mkdir(parents=True, exist_ok=True)

    best_t1 = max(records, key=lambda record: float(record.test_t1["score"]))
    best_t3 = max(records, key=lambda record: float(record.test_t3["score"]))

    copy_model_snapshot(Path(best_t1.checkpoint_path), best_dir / "best_test1_t1")
    write_json(
        best_dir / "best_test1_t1.metadata.json",
        {
            "checkpoint_name": best_t1.checkpoint_name,
            "checkpoint_path": best_t1.checkpoint_path,
            "global_step": best_t1.global_step,
            "metrics": best_t1.test_t1,
        },
    )

    copy_model_snapshot(Path(best_t3.checkpoint_path), best_dir / "best_test1_t3")
    write_json(
        best_dir / "best_test1_t3.metadata.json",
        {
            "checkpoint_name": best_t3.checkpoint_name,
            "checkpoint_path": best_t3.checkpoint_path,
            "global_step": best_t3.global_step,
            "metrics": best_t3.test_t3,
        },
    )


def metric_bar(value: float, width: int = 30) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "#" * filled + "." * (width - filled)


def format_step(record: EvaluationRecord) -> str:
    if record.global_step is not None:
        return str(record.global_step)
    return record.checkpoint_name


def print_records_summary(records: list[EvaluationRecord]) -> None:
    if not records:
        return

    ordered = sorted(
        records,
        key=lambda record: (record.global_step is None, record.global_step if record.global_step is not None else 10**18),
    )

    print("")
    print("Checkpoint scores")
    print(
        f"{'step':>10}  {'t1_score':>10}  {'t1_abs':>10}  {'t1_full':>10}  {'t3_score':>10}"
    )
    print("-" * 60)
    for record in ordered:
        print(
            f"{format_step(record):>10}  "
            f"{float(record.test_t1['score']):>10.6f}  "
            f"{float(record.test_t1['abstracts_macro_f1']):>10.6f}  "
            f"{float(record.test_t1['full_texts_macro_f1']):>10.6f}  "
            f"{float(record.test_t3['score']):>10.6f}"
        )

    print("")
    print("ASCII charts")
    for record in ordered:
        t1_value = float(record.test_t1["score"])
        t3_value = float(record.test_t3["score"])
        print(f"{format_step(record):>10}  t1 {metric_bar(t1_value)} {t1_value:.6f}")
        print(f"{'':>10}  t3 {metric_bar(t3_value)} {t3_value:.6f}")

    best_t1 = max(ordered, key=lambda record: float(record.test_t1["score"]))
    best_t3 = max(ordered, key=lambda record: float(record.test_t3["score"]))
    print("")
    print(
        f"Best test1_t1: step={format_step(best_t1)} score={float(best_t1.test_t1['score']):.6f} "
        f"(abs={float(best_t1.test_t1['abstracts_macro_f1']):.6f}, full={float(best_t1.test_t1['full_texts_macro_f1']):.6f})"
    )
    print(f"Best test1_t3: step={format_step(best_t3)} score={float(best_t3.test_t3['score']):.6f}")


def run_training(generated_config_path: Path, device_name: str, single_gpu: bool) -> subprocess.Popen[bytes]:
    env = build_child_env(device_name=device_name, single_gpu=single_gpu)
    command = [
        sys.executable,
        str(BINDER_ROOT / "run_ner.py"),
        str(generated_config_path),
    ]
    return subprocess.Popen(command, cwd=BINDER_ROOT, env=env)


def monitor_training_and_score(
    output_dir: Path,
    launcher_dir: Path,
    generated_config_path: Path,
    args: argparse.Namespace,
) -> list[EvaluationRecord]:
    device_name = resolve_device_name(args.device)
    records_path = launcher_dir / RECORDS_FILENAME
    records = load_existing_records(records_path)
    seen = {record.checkpoint_name for record in records}

    process: subprocess.Popen[bytes] | None = None
    if not args.skip_train:
        process = run_training(
            generated_config_path=generated_config_path,
            device_name=device_name,
            single_gpu=args.single_gpu,
        )

    try:
        while True:
            had_new_record = False
            for checkpoint_name, checkpoint_path in iter_candidate_model_dirs(output_dir, include_final=False):
                if checkpoint_name in seen:
                    continue
                try:
                    record = evaluate_checkpoint(
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=checkpoint_path,
                        generated_config_path=generated_config_path,
                        output_dir=output_dir,
                        test_t1_file=args.test_t1_file.expanduser().resolve(),
                        test_t3_file=args.test_t3_file.expanduser().resolve(),
                        threshold=args.threshold,
                        device_name=device_name,
                        single_gpu=args.single_gpu,
                    )
                except Exception as error:
                    print(f"Checkpoint {checkpoint_name} is not ready for scoring yet: {error}")
                    continue
                records.append(record)
                append_record(records_path, record)
                seen.add(checkpoint_name)
                maybe_update_best_snapshots(output_dir, records)
                print_records_summary(records)
                had_new_record = True

            if process is None:
                break

            return_code = process.poll()
            if return_code is not None:
                if not had_new_record:
                    for checkpoint_name, checkpoint_path in iter_candidate_model_dirs(output_dir, include_final=True):
                        if checkpoint_name in seen:
                            continue
                        record = evaluate_checkpoint(
                            checkpoint_name=checkpoint_name,
                            checkpoint_path=checkpoint_path,
                            generated_config_path=generated_config_path,
                            output_dir=output_dir,
                            test_t1_file=args.test_t1_file.expanduser().resolve(),
                            test_t3_file=args.test_t3_file.expanduser().resolve(),
                            threshold=args.threshold,
                            device_name=device_name,
                            single_gpu=args.single_gpu,
                        )
                        records.append(record)
                        append_record(records_path, record)
                        seen.add(checkpoint_name)
                    maybe_update_best_snapshots(output_dir, records)
                    if records:
                        print_records_summary(records)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, process.args)
                break

            time.sleep(max(1.0, args.poll_seconds))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()

    return records


def main() -> None:
    args = build_parser().parse_args()

    if not args.base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config_path}")
    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not args.test_t1_file.exists():
        raise FileNotFoundError(f"test1_t1 file not found: {args.test_t1_file}")
    if not args.test_t3_file.exists():
        raise FileNotFoundError(f"test1_t3 file not found: {args.test_t3_file}")

    base_config = load_json(args.base_config_path)
    output_dir = resolve_output_dir(base_config, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher_dir = output_dir.parent / f"{output_dir.name}{LAUNCHER_DIRNAME}"
    launcher_dir.mkdir(parents=True, exist_ok=True)

    generated_config = build_generated_config(base_config, args, output_dir)
    generated_config_path = launcher_dir / GENERATED_CONFIG_FILENAME
    write_json(generated_config_path, generated_config)

    effective_train_batch = (
        int(generated_config["per_device_train_batch_size"])
        * resolve_num_devices(resolve_device_name(args.device), single_gpu=args.single_gpu)
        * int(generated_config["gradient_accumulation_steps"])
    )
    print("Generated train config:", generated_config_path)
    print("Output dir:", output_dir)
    print("Per-device train batch size:", generated_config["per_device_train_batch_size"])
    print("Gradient accumulation steps:", generated_config["gradient_accumulation_steps"])
    print("Effective global train batch size:", effective_train_batch)
    print("Training file:", generated_config["train_file"])
    print("test1_t1 file:", args.test_t1_file.expanduser().resolve())
    print("test1_t3 file:", args.test_t3_file.expanduser().resolve())

    records = monitor_training_and_score(
        output_dir=output_dir,
        launcher_dir=launcher_dir,
        generated_config_path=generated_config_path,
        args=args,
    )
    maybe_update_best_snapshots(output_dir, records)
    if records:
        print_records_summary(records)
    else:
        print("No checkpoints were scored.")


if __name__ == "__main__":
    main()
