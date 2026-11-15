from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fake_review_detection.config import load_config
from fake_review_detection.explainability import explain_stylometric_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["stylometric"], default="stylometric")
    args = parser.parse_args()

    config = load_config()

    if args.model == "stylometric":
        feature_path = config["stylometric"]["features_path"]
        model_path = config["stylometric"]["model_path"]

        features = pd.read_csv(feature_path)
        explain_stylometric_model(
            model_path=model_path,
            feature_frame=features,
            output_dir=config["results"]["shap_dir"],
        )
        print("Saved stylometric SHAP plots.")


if __name__ == "__main__":
    main()
