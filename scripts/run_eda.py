from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fake_review_detection.config import load_config


def main():
    config = load_config()
    splits_dir = Path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")

    train_df["char_length"] = train_df["text"].str.len()
    train_df["word_length"] = train_df["text"].str.split().apply(len)

    plots_dir = Path(config["results"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    train_df["label"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_class_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    train_df.boxplot(column="word_length", by="label", ax=ax)
    ax.set_title("Review Length by Class")
    ax.set_ylabel("Word Count")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_word_length_by_class.png", dpi=200)
    plt.close(fig)

    print("EDA plots saved.")


if __name__ == "__main__":
    main()
