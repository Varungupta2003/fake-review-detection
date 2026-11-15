import pytest

from fake_review_detection.evaluation import compute_classification_metrics


def test_metrics_on_known_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.7, 0.9]
    m = compute_classification_metrics(y_true, y_pred, y_score)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["pr_auc"] == pytest.approx(1.0)
