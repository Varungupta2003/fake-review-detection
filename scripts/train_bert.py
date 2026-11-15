from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from transformers import Trainer

from fake_review_detection.config import load_config
from fake_review_detection.evaluation import save_confusion_matrix_plot, save_pr_curve
from fake_review_detection.models.bert_dataset import ReviewTextDataset
from fake_review_detection.models.bert_model import (
    build_training_arguments,
    compute_metrics,
    load_bert_artifacts,
    tokenize_texts,
)
from fake_review_detection.utils import save_json, set_seed


def main():
    config = load_config()
    set_seed(config["project"]["random_seed"])

    train_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "val.csv")
    test_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "test.csv")

    artifacts = load_bert_artifacts(config["bert"]["pretrained_model_name"])

    train_enc = tokenize_texts(artifacts.tokenizer, train_df["text"], config["data"]["max_length"])
    val_enc = tokenize_texts(artifacts.tokenizer, val_df["text"], config["data"]["max_length"])
    test_enc = tokenize_texts(artifacts.tokenizer, test_df["text"], config["data"]["max_length"])

    train_ds = ReviewTextDataset(train_enc, train_df["label"].tolist())
    val_ds = ReviewTextDataset(val_enc, val_df["label"].tolist())
    test_ds = ReviewTextDataset(test_enc, test_df["label"].tolist())

    args = build_training_arguments(
        output_dir=config["bert"]["model_dir"],
        epochs=config["bert"]["epochs"],
        batch_size=config["bert"]["batch_size"],
        learning_rate=float(config["bert"]["learning_rate"]),
    )

    trainer_kwargs = {
        "model": artifacts.model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }
    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = artifacts.tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_score = probs[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_true = test_df["label"].values

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }

    save_json(metrics, Path(config["results"]["metrics_dir"]) / "bert_metrics.json")
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }).to_csv(Path(config["results"]["metrics_dir"]) / "bert_test_predictions.csv", index=False)

    save_confusion_matrix_plot(
        y_true,
        y_pred,
        Path(config["results"]["plots_dir"]) / "bert_confusion_matrix.png",
        "BERT Confusion Matrix",
    )
    save_pr_curve(
        y_true,
        y_score,
        Path(config["results"]["plots_dir"]) / "bert_pr_curve.png",
        "BERT PR Curve",
    )

    trainer.save_model(config["bert"]["model_dir"])
    artifacts.tokenizer.save_pretrained(config["bert"]["model_dir"])

    print(metrics)


if __name__ == "__main__":
    main()
