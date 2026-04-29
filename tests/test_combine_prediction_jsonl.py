import json

from common.combine_prediction_jsonl import combine_prediction_files


def test_combine_prediction_files_normalizes_labels_and_drops_extra_fields(tmp_path) -> None:
    input_dir = tmp_path / "predictions"
    input_dir.mkdir()

    file_payloads = {
        "test2_t3_v2.jsonl": [
            {"id": "a", "text": "alpha", "label": [[0, 5]], "extra": 1},
        ],
        "test2_t12_v2.jsonl": [
            {"id": "b", "text": "beta", "label": [[1, 3, "common"]], "candidates": [[1, 3]]},
        ],
        "test_all_domains_v2.jsonl": [
            {"id": "c", "text": "gamma", "label": []},
        ],
    }

    for filename, rows in file_payloads.items():
        (input_dir / filename).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    output_path = tmp_path / "combined.jsonl"
    row_count = combine_prediction_files(input_dir=input_dir, output_path=output_path)

    assert row_count == 3

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {"id": "a", "text": "alpha", "label": [[0, 5, "specific"]]},
        {"id": "b", "text": "beta", "label": [[1, 3, "common"]]},
        {"id": "c", "text": "gamma", "label": []},
    ]
