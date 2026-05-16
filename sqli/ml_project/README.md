# SQL Injection Detection - Logistic Regression From Scratch

This project implements a modular, pure-Python SQL injection detector. It does not require `numpy`, `pandas`, or `scikit-learn`.

The pipeline follows the requested plan:

1. SQL-aware preprocessing that lowercases, decodes HTML entities, and preserves SQL-relevant symbols.
2. Feature extraction with TF-IDF unigrams/bigrams plus handcrafted SQL features.
3. Stratified 80/20 train-test split, with an inner validation split for tuning.
4. Class-weighted binary cross-entropy for the mildly imbalanced dataset.
5. Logistic regression implemented from scratch with sigmoid, BCE loss, gradient descent, and L2 regularization.
6. Threshold tuning focused on recall using F2 score, with a threshold sweep saved for review.
7. Evaluation with precision, recall, F1, specificity, accuracy, and confusion matrix.

## Project Structure

- `sql_injection_lr/preprocessing.py`: cleaning and tokenization
- `sql_injection_lr/features.py`: TF-IDF and manual SQL features
- `sql_injection_lr/model.py`: from-scratch logistic regression
- `sql_injection_lr/split.py`: stratified splitting
- `sql_injection_lr/metrics.py`: evaluation and threshold metrics
- `sql_injection_lr/tuning.py`: simple hyperparameter and threshold tuning
- `sql_injection_lr/pipeline.py`: save/load prediction pipeline
- `train.py`: train, tune, evaluate, and save artifacts
- `predict.py`: classify a new query with the saved model

## Run Training

```bash
python train.py
```

Outputs:

- `artifacts/sqli_logreg_model.json`
- `reports/evaluation.json`

## Predict

```bash
python predict.py "a' or 1 = 1; --"
python predict.py "show me today's account balance"
```
