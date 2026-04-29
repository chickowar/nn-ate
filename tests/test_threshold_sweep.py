from common.threshold_sweep import resolve_thresholds, select_spans


def test_select_spans_uses_threshold_inclusively() -> None:
    candidates = [(0, 4), (5, 9), (10, 14)]
    probabilities = [0.49, 0.5, 0.9]

    spans = select_spans(candidates=candidates, probabilities=probabilities, threshold=0.5)

    assert spans == [[5, 9], [10, 14]]


def test_resolve_thresholds_builds_uniform_sorted_grid() -> None:
    class Args:
        thresholds = None
        threshold_count = 5
        threshold_min = 0.1
        threshold_max = 0.9

    resolved = resolve_thresholds(Args())

    assert resolved == [0.1, 0.3, 0.5, 0.7, 0.9]
