from __future__ import annotations

from pathlib import Path

from fake_review_detection.config import load_config
from fake_review_detection.stats import mcnemar_from_prediction_files
from fake_review_detection.utils import save_json


def main():
    config = load_config()
    metrics_dir = Path(config["results"]["metrics_dir"])

    comparisons = {
        "stylometric_vs_bert": (
            metrics_dir / "stylometric_test_predictions.csv",
            metrics_dir / "bert_test_predictions.csv",
        ),
        "stylometric_vs_hybrid": (
            metrics_dir / "stylometric_test_predictions.csv",
            metrics_dir / "hybrid_test_predictions.csv",
        ),
        "bert_vs_hybrid": (
            metrics_dir / "bert_test_predictions.csv",
            metrics_dir / "hybrid_test_predictions.csv",
        ),
    }

    results = {}
    for name, (file_a, file_b) in comparisons.items():
        if file_a.exists() and file_b.exists():
            results[name] = mcnemar_from_prediction_files(file_a, file_b)

    save_json(results, metrics_dir / "mcnemar_results.json")
    print(results)


if __name__ == "__main__":
    main()
