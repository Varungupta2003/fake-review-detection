import pandas as pd
import pytest

from fake_review_detection.stats import mcnemar_from_prediction_files


def _preds(tmp_path, name, y_true, y_pred):
    path = tmp_path / name
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)
    return path


def test_mcnemar_counts_discordant_pairs(tmp_path):
    y_true = [1, 1, 1, 1, 0, 0]
    a = _preds(tmp_path, "a.csv", y_true, [1, 0, 0, 1, 0, 0])  # wrong on rows 1 and 2
    b = _preds(tmp_path, "b.csv", y_true, [1, 1, 0, 0, 0, 0])  # wrong on rows 2 and 3
    result = mcnemar_from_prediction_files(a, b)
    assert result["n01"] == 1  # a wrong, b right (row 1)
    assert result["n10"] == 1  # a right, b wrong (row 3)
    assert 0.0 <= result["p_value"] <= 1.0


def test_identical_models_are_not_significant(tmp_path):
    y_true = [1, 0, 1, 0]
    a = _preds(tmp_path, "a.csv", y_true, [1, 0, 0, 0])
    b = _preds(tmp_path, "b.csv", y_true, [1, 0, 0, 0])
    result = mcnemar_from_prediction_files(a, b)
    assert result["n01"] == 0 and result["n10"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_mismatched_ground_truth_raises(tmp_path):
    a = _preds(tmp_path, "a.csv", [1, 0], [1, 0])
    b = _preds(tmp_path, "b.csv", [0, 1], [1, 0])
    with pytest.raises(ValueError):
        mcnemar_from_prediction_files(a, b)
