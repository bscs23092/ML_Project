# CSRF Boosting Model

This folder trains a leakage-safe boosting model on the Mitch `features_matrix.csv` dataset.

## Training

```powershell
python csrf\train.py
```

The script:

- maps `flag == "y"` to `1` and `n/u/m` to `0`
- drops `reqId` and `flag`
- uses a sklearn `Pipeline` for log transforms, variance filtering, and boosting
- uses a stratified 80/20 hold-out test split
- tunes only on the training split with stratified 5-fold cross validation
- selects the decision threshold from out-of-fold training predictions
- saves the best parameters to `csrf/artifacts/best_params.json`
- saves the best model artifact to `csrf/artifacts/csrf_boosting_pipeline.joblib`
- saves evaluation outputs to `csrf/reports/`

## Prediction

```powershell
python csrf\predict.py --json-file path\to\feature_row.json
```

The JSON row must contain the same feature columns as the Mitch CSV, except `reqId` and `flag`.

