from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_stylometric_classifier(X_train: pd.DataFrame, y_train):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_stylometric_classifier(model, model_path: str) -> None:
    joblib.dump(model, model_path)
