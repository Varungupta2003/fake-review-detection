from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def explain_stylometric_model(
    model_path: str,
    feature_frame: pd.DataFrame,
    output_dir: str,
    max_display: int = 15,
) -> None:
    model = joblib.load(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, feature_frame)
    shap_values = explainer(feature_frame)

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "stylometric_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "stylometric_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
