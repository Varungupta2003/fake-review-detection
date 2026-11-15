from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    label_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # The original Ott et al. CSV names its label column "deceptive".
    if label_column not in df.columns and "deceptive" in df.columns:
        df = df.rename(columns={"deceptive": label_column})
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Dataset must contain columns '{text_column}' and '{label_column}'. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[text_column, label_column]].copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].dropna()

    if label_map is None:
        label_map = {"truthful": 0, "deceptive": 1}

    if not pd.api.types.is_numeric_dtype(df[label_column]):
        normalized = df[label_column].astype(str).str.strip().str.lower()
        if set(normalized.unique()).issubset(set(label_map.keys())):
            df[label_column] = normalized.map(label_map)
        else:
            raise ValueError(
                "String labels detected, but they do not match the expected values: "
                f"{list(label_map.keys())}"
            )

    df[label_column] = df[label_column].astype(int)
    df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    return df.rename(columns={text_column: "text", label_column: "label"})


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
