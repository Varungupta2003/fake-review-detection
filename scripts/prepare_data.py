from __future__ import annotations

from pathlib import Path

from fake_review_detection.config import load_config
from fake_review_detection.data import load_dataset, save_splits, stratified_split
from fake_review_detection.preprocess import clean_text
from fake_review_detection.utils import set_seed


def main():
    config = load_config()
    set_seed(config["project"]["random_seed"])

    df = load_dataset(
        csv_path=config["data"]["raw_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        label_map=config["labels"],
    )
    df["text"] = df["text"].apply(clean_text)

    train_df, val_df, test_df = stratified_split(
        df,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["project"]["random_seed"],
    )

    save_splits(train_df, val_df, test_df, config["data"]["splits_dir"])

    print("Saved dataset splits:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")


if __name__ == "__main__":
    main()
