# Fake Review Detection: Stylometry vs BERT vs Hybrid

[![CI](https://github.com/Varungupta2003/fake-review-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Varungupta2003/fake-review-detection/actions)

Detecting deceptive hotel reviews with three models, compared on the same held-out split:

1. **Stylometric baseline**: readability, lexical and POS features (textstat + spaCy) with logistic regression.
2. **BERT**: `bert-base-uncased` fine-tuned for binary classification.
3. **Hybrid**: BERT `[CLS]` embedding (768-d) concatenated with the stylometric features, fed to an MLP (early fusion).

Built as part of my MSc in Artificial Intelligence (University of East London, 2026).

## Results (held-out test set, 240 reviews)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Stylometric + LogReg | 0.767 | 0.722 | 0.867 | 0.788 | 0.800 |
| BERT (fine-tuned) | 0.867 | 0.819 | 0.942 | 0.876 | 0.965 |
| Hybrid (BERT + stylometric) | 0.900 | 0.921 | 0.875 | 0.897 | 0.970 |

McNemar's test (continuity-corrected) on paired predictions:

| Comparison | Discordant pairs (A wrong/B right, A right/B wrong) | p-value |
|---|---|---|
| Stylometric vs BERT | 32 / 8 | 0.0003 |
| Stylometric vs Hybrid | 45 / 13 | <0.0001 |
| BERT vs Hybrid | 16 / 8 | 0.153 (not significant) |

**How to read this honestly**
- Both neural models beat the stylometric baseline significantly. The hybrid's edge over plain BERT is **not** statistically significant on this test set.
- The test set is small (240 reviews), so each accuracy figure has a wide confidence interval (roughly ±4 points).
- The corpus is the Ott et al. benchmark: deceptive reviews were written by paid crowd workers, and the truthful ones were collected from a review site. High scores here do **not** show the models would catch real-world fake reviews.
- The stylometric and BERT figures are re-checkable from the committed prediction CSVs. The hybrid's metrics come from the saved JSON; its prediction file was not kept, so the two McNemar comparisons involving the hybrid cannot be re-run from this repo.

See [docs/RESULTS.md](docs/RESULTS.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for detail.

## Repository layout

```
src/fake_review_detection/   package: data, preprocessing, features, models, evaluation, stats, explainability
scripts/                     entry points: prepare_data, train_*, run_eda, run_stats, run_explainability
tests/                       pytest suite (data loading/splitting, metrics, McNemar)
results/metrics, plots       saved metrics JSON, test predictions, confusion matrices, PR curves
docs/                        methodology and results write-up
app.py                       Streamlit demo (live prediction needs trained models; results tabs work without)
config.yaml                  paths and hyperparameters
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m spacy download en_core_web_sm

# 1. put the dataset at data/raw/ott_deceptive_reviews.csv  (see data/README.md)
python scripts/prepare_data.py          # clean + stratified 70/15/15 split (seed 42)
python scripts/train_stylometric.py
python scripts/train_bert.py            # ~50 min on the CPU used originally; a GPU is much faster
python scripts/train_hybrid.py
python scripts/run_stats.py             # McNemar tests
python scripts/run_explainability.py    # SHAP on the stylometric model
streamlit run app.py                    # optional demo
```

Run the tests with `pytest` (they need only pandas, scikit-learn, scipy and statsmodels, not torch).

## Limitations

- Single dataset, single split, single seed: no cross-validation and no variance across runs.
- Domain is hotel reviews only; no out-of-domain test.
- Trained models (about 4 GB) are not in the repo; retrain with the scripts above.
- The dataset is not redistributed; see `data/README.md`.

## Licence and citation

MIT licence. Dataset: Ott, Choi, Cardie and Hancock (2011), *Finding Deceptive Opinion Spam by Any Stretch of the Imagination*, ACL.
