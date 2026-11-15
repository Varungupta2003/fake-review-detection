from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from fake_review_detection.config import load_config
from fake_review_detection.evaluation import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_pr_curve,
)
from fake_review_detection.features import StylometricFeatureExtractor
from fake_review_detection.models.bert_model import load_bert_artifacts, tokenize_texts
from fake_review_detection.models.hybrid_dataset import HybridReviewDataset
from fake_review_detection.models.hybrid_model import HybridBertStylometricClassifier
from fake_review_detection.trainers import (
    evaluate_hybrid_model,
    measure_inference_latency,
    train_hybrid_model,
)
from fake_review_detection.utils import save_json, set_seed


def main():
    config = load_config()
    set_seed(config["project"]["random_seed"])

    train_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "val.csv")
    test_df = pd.read_csv(Path(config["data"]["splits_dir"]) / "test.csv")

    extractor = StylometricFeatureExtractor()
    X_train_style = extractor.transform(train_df["text"])
    X_val_style = extractor.transform(val_df["text"])
    X_test_style = extractor.transform(test_df["text"])

    scaler = StandardScaler()
    X_train_style = scaler.fit_transform(X_train_style)
    X_val_style = scaler.transform(X_val_style)
    X_test_style = scaler.transform(X_test_style)

    artifacts = load_bert_artifacts(config["bert"]["pretrained_model_name"])

    train_enc = tokenize_texts(artifacts.tokenizer, train_df["text"], config["data"]["max_length"])
    val_enc = tokenize_texts(artifacts.tokenizer, val_df["text"], config["data"]["max_length"])
    test_enc = tokenize_texts(artifacts.tokenizer, test_df["text"], config["data"]["max_length"])

    train_ds = HybridReviewDataset(train_enc, X_train_style, train_df["label"].tolist())
    val_ds = HybridReviewDataset(val_enc, X_val_style, val_df["label"].tolist())
    test_ds = HybridReviewDataset(test_enc, X_test_style, test_df["label"].tolist())

    batch_size = config["hybrid"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = HybridBertStylometricClassifier(
        bert_model_name=config["bert"]["pretrained_model_name"],
        num_stylometric_features=X_train_style.shape[1],
        hidden_dim=config["hybrid"]["hidden_dim"],
        dropout=float(config["hybrid"]["dropout"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_info = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config["hybrid"]["epochs"],
        learning_rate=float(config["hybrid"]["learning_rate"]),
    )
    model = train_info["model"]

    test_metrics = evaluate_hybrid_model(model, test_loader, device)
    latency = measure_inference_latency(model, test_loader, device)

    metrics = compute_classification_metrics(
        test_metrics["y_true"], test_metrics["y_pred"], test_metrics["y_score"]
    )
    metrics["training_time_seconds"] = train_info["training_time_seconds"]
    metrics["memory_delta_mb"] = train_info["memory_delta_mb"]
    metrics["avg_inference_latency_seconds_per_batch"] = latency

    save_json(metrics, Path(config["results"]["metrics_dir"]) / "hybrid_metrics.json")
    pd.DataFrame({
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "y_score": test_metrics["y_score"],
    }).to_csv(Path(config["results"]["metrics_dir"]) / "hybrid_test_predictions.csv", index=False)

    torch.save(model.state_dict(), Path(config["hybrid"]["model_dir"]) / "hybrid_model.pt")

    save_confusion_matrix_plot(
        test_metrics["y_true"], test_metrics["y_pred"],
        Path(config["results"]["plots_dir"]) / "hybrid_confusion_matrix.png",
        "Hybrid Confusion Matrix",
    )
    save_pr_curve(
        test_metrics["y_true"], test_metrics["y_score"],
        Path(config["results"]["plots_dir"]) / "hybrid_pr_curve.png",
        "Hybrid PR Curve",
    )

    print(metrics)


if __name__ == "__main__":
    main()
