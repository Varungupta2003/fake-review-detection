from __future__ import annotations

from pathlib import Path

import pandas as pd

from fake_review_detection.config import load_config
from fake_review_detection.evaluation import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_pr_curve,
)
from fake_review_detection.features import StylometricFeatureExtractor
from fake_review_detection.models.stylometric_model import (
    save_stylometric_classifier,
    train_stylometric_classifier,
)
from fake_review_detection.utils import Timer, save_json, set_seed


def main():
    config = load_config()
    set_seed(config["project"]["random_seed"])

    train_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "val.csv")
    test_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "test.csv")

    extractor = StylometricFeatureExtractor()

    X_train = extractor.transform(train_df["text"])
    X_val = extractor.transform(val_df["text"])
    X_test = extractor.transform(test_df["text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    with Timer() as train_timer:
        model = train_stylometric_classifier(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    val_preds = (val_scores >= 0.5).astype(int)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_preds = (test_scores >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_test, test_preds, test_scores)
    metrics["training_time_seconds"] = train_timer.elapsed

    save_stylometric_classifier(model, config["stylometric"]["model_path"])
    X_test.to_csv(config["stylometric"]["features_path"], index=False)

    save_json(metrics, Path(config["results"]["metrics_dir"]) / "stylometric_metrics.json")
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": test_preds,
        "y_score": test_scores,
    }).to_csv(Path(config["results"]["metrics_dir"]) / "stylometric_test_predictions.csv", index=False)

    save_confusion_matrix_plot(
        y_test, test_preds,
        Path(config["results"]["plots_dir"]) / "stylometric_confusion_matrix.png",
        "Stylometric Logistic Regression Confusion Matrix",
    )
    save_pr_curve(
        y_test, test_scores,
        Path(config["results"]["plots_dir"]) / "stylometric_pr_curve.png",
        "Stylometric Logistic Regression PR Curve",
    )

    print("Validation F1-ready stylometric baseline trained.")
    print(metrics)


if __name__ == "__main__":
    main()
