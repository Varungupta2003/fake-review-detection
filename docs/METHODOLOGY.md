# Methodology

## Data
Ott et al. Deceptive Opinion Spam Corpus: 1,600 hotel reviews (800 truthful, 800 deceptive). Empty and duplicate texts are removed, then a stratified 70/15/15 train/validation/test split is made with seed 42 (`scripts/prepare_data.py`). The test set has 240 reviews (120 per class). Models are trained on train, tuned/early-checked on validation, and scored once on test.

## Models
**Stylometric baseline.** Features from `StylometricFeatureExtractor` (textstat readability scores, sentence and word length statistics, lexical diversity, punctuation rates, POS-tag proportions via spaCy `en_core_web_sm`). Standardised, then logistic regression.

**BERT.** `bert-base-uncased`, max length 256, batch size 16, learning rate 2e-5, 3 epochs, weight decay 0.01, 10% warmup.

**Hybrid.** Early fusion: BERT `[CLS]` vector (768-d) concatenated with the scaled stylometric vector, passed through an MLP (hidden dim 256, dropout 0.2). Same optimiser settings as BERT.

## Evaluation
Accuracy, precision, recall, F1 and PR-AUC with "deceptive" as the positive class. Significance: McNemar's test with continuity correction on paired test predictions (`statsmodels`).

## Explainability
SHAP values on the stylometric logistic regression to see which features push predictions towards "deceptive". SHAP was not applied to BERT or the hybrid.

## Known gaps
No cross-validation or multiple seeds; no hyperparameter search reported; no out-of-domain evaluation.
