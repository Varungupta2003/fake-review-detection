from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except Exception:
    sm_mcnemar = None


def mcnemar_from_prediction_files(
    file_a: str | Path,
    file_b: str | Path,
    truth_column: str = "y_true",
    pred_column: str = "y_pred",
) -> dict[str, float]:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    if not all(df_a[truth_column] == df_b[truth_column]):
        raise ValueError("Ground-truth labels do not match between prediction files.")

    a_correct = df_a[pred_column] == df_a[truth_column]
    b_correct = df_b[pred_column] == df_b[truth_column]

    n01 = int(((a_correct == 0) & (b_correct == 1)).sum())
    n10 = int(((a_correct == 1) & (b_correct == 0)).sum())

    if n01 + n10 == 0:
        statistic, p_value = 0.0, 1.0
    elif sm_mcnemar is not None:
        table = [[0, n01], [n10, 0]]
        result = sm_mcnemar(table, exact=False, correction=True)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    else:
        total = n01 + n10
        if total == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = float((abs(n01 - n10) - 1) ** 2 / total)
            p_value = float(binomtest(min(n01, n10), n=total, p=0.5, alternative="two-sided").pvalue)

    return {
        "n01": n01,
        "n10": n10,
        "statistic": statistic,
        "p_value": p_value,
    }
