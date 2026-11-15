import pandas as pd
import pytest

from fake_review_detection.data import stratified_split


def _frame(n=200):
    return pd.DataFrame({
        "text": [f"sample {i}" for i in range(n)],
        "label": [0] * (n // 2) + [1] * (n // 2),
    })


def test_split_sizes_and_no_overlap():
    train_df, val_df, test_df = stratified_split(_frame(), test_size=0.15, val_size=0.15, random_state=42)
    assert len(train_df) + len(val_df) + len(test_df) == 200
    assert len(test_df) == 30 and abs(len(val_df) - 30) <= 1
    all_texts = pd.concat([train_df, val_df, test_df])["text"]
    assert all_texts.is_unique


def test_split_is_stratified():
    for part in stratified_split(_frame(), test_size=0.15, val_size=0.15, random_state=42):
        assert part["label"].mean() == pytest.approx(0.5, abs=0.05)


def test_split_is_reproducible():
    a = stratified_split(_frame(), random_state=7)[2]
    b = stratified_split(_frame(), random_state=7)[2]
    assert a["text"].tolist() == b["text"].tolist()


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        stratified_split(_frame(), test_size=0.6, val_size=0.4)
