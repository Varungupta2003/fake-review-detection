# Results

Test split: 240 reviews, 120 truthful and 120 deceptive.

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Stylometric + LogReg | 0.7667 | 0.7222 | 0.8667 | 0.7879 | 0.7995 |
| BERT | 0.8667 | 0.8188 | 0.9417 | 0.8760 | 0.9654 |
| Hybrid | 0.9000 | 0.9211 | 0.8750 | 0.8974 | 0.9702 |

Confusion matrices (rows = truth truthful/deceptive, columns = predicted), recomputed from the committed prediction CSVs:

- Stylometric: [[80, 40], [16, 104]]
- BERT: [[95, 25], [7, 113]]

Plots are in `results/plots/` (confusion matrix and PR curve for the stylometric model and BERT).

## McNemar's test

| Pair | n01 | n10 | statistic | p-value |
|---|---|---|---|---|
| Stylometric vs BERT | 32 | 8 | 13.23 | 0.00028 |
| Stylometric vs Hybrid | 45 | 13 | 16.57 | 0.000047 |
| BERT vs Hybrid | 16 | 8 | 2.04 | 0.153 |

n01 = first model wrong and second right; n10 = first right and second wrong. The Stylometric-vs-BERT row was re-derived from the prediction CSVs in this repo. The two rows involving the hybrid come from the original run's `mcnemar_results.json`; the hybrid prediction file was not kept, so they cannot be re-derived here (they are consistent with the accuracies: 184 + 45 − 13 = 216 = 0.90 × 240).

## Caveats
- The hybrid's F1 advantage over BERT (0.897 vs 0.876) is not statistically significant (p = 0.15).
- 240 test reviews means wide uncertainty; one seed, one split.
- Ott et al. deceptive reviews are crowd-sourced, so results may not transfer to real platforms.
- BERT recall (0.94) is higher than the hybrid's (0.875) while the hybrid is more precise; which is "better" depends on whether false negatives or false positives matter more.
