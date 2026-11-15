import pandas as pd
import pytest

from fake_review_detection.data import load_dataset, stratified_split


def _write(tmp_path, df):
    path = tmp_path / "reviews.csv"
    df.to_csv(path, index=False)
    return path


def test_string_labels_are_mapped_and_duplicates_dropped(tmp_path):
    df = pd.DataFrame({
        "text": ["great room", "great room", "  awful stay  ", ""],
        "label": ["truthful", "truthful", "deceptive", "truthful"],
    })
    out = load_dataset(_write(tmp_path, df))
    assert list(out.columns) == ["text", "label"]
    assert out["text"].tolist() == ["great room", "awful stay"]
    assert out["label"].tolist() == [0, 1]


def test_original_ott_label_column_name_is_accepted(tmp_path):
    df = pd.DataFrame({
        "deceptive": ["truthful", "deceptive"],
        "hotel": ["a", "b"],
        "text": ["one", "two"],
    })
    out = load_dataset(_write(tmp_path, df))
    assert out["label"].tolist() == [0, 1]


def test_unknown_string_label_raises(tmp_path):
    df = pd.DataFrame({"text": ["x"], "label": ["maybe"]})
    with pytest.raises(ValueError):
        load_dataset(_write(tmp_path, df))


def test_missing_columns_raise(tmp_path):
    df = pd.DataFrame({"review": ["x"], "label": [0]})
    with pytest.raises(ValueError):
        load_dataset(_write(tmp_path, df))
