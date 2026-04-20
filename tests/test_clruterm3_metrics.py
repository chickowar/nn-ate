import json

from term_datasets.CL_RuTerm3 import compute_exact_match_metrics


def test_compute_exact_match_metrics_with_labels() -> None:
    gold = [
        {"id": "doc-1", "label": [[0, 4, "specific"], [10, 14, "common"]]},
        {"id": "doc-2", "label": [[2, 7, "nomen"]]},
    ]
    predicted = [
        {"id": "doc-1", "label": [[10, 14, "common"], [0, 4, "specific"]]},
        {"id": "doc-2", "label": [[2, 7, "specific"]]},
    ]

    metrics = compute_exact_match_metrics(gold=gold, predicted=predicted, use_labels=True)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["f1"] == 2 / 3


def test_compute_exact_match_metrics_without_labels() -> None:
    gold = [{"id": "doc-1", "label": [[0, 4, "specific"], [10, 14, "common"]]}]
    predicted = [{"id": "doc-1", "label": [[10, 14, "nomen"], [0, 4, "common"]]}]

    metrics = compute_exact_match_metrics(gold=gold, predicted=predicted, use_labels=False)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["f1"] == 1.0


def test_compute_exact_match_metrics_from_jsonl_paths(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    predicted_path = tmp_path / "pred.jsonl"

    gold_rows = [
        {"id": "doc-1", "label": [[0, 4, "specific"]]},
        {"id": "doc-2", "label": [[5, 8, "common"]]},
    ]
    predicted_rows = [
        {"id": "doc-2", "label": [[5, 8, "common"]]},
        {"id": "doc-1", "label": [[0, 4, "specific"]]},
    ]

    gold_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in gold_rows) + "\n",
        encoding="utf-8",
    )
    predicted_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predicted_rows) + "\n",
        encoding="utf-8",
    )

    metrics = compute_exact_match_metrics(gold=gold_path, predicted=predicted_path, use_labels=True)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
