# Web Attack Detection

Unified Streamlit app for the three project modules:

- XSS detection with the CNN model in `xss/`
- SQL injection detection with the from-scratch logistic regression pipeline in `sqli/ml_project/`
- CSRF relevance detection with the boosting pipeline in `csrf/`

The original module folders are still kept separate for training, evaluation, datasets, and saved artifacts. The root files are the shared entry point for running the combined app and installing dependencies.

## Project Structure

- `app.py`: Unified Streamlit UI. Choose `XSS`, `SQL Injection`, or `CSRF` from the sidebar.
- `requirements.txt`: Combined dependencies for both modules.
- `xss/`: XSS dataset, model code, training scripts, evaluation scripts, and saved CNN artifacts.
- `sqli/ml_project/`: SQLi dataset, from-scratch logistic regression package, training script, prediction script, and saved artifacts.
- `csrf/`: Mitch CSRF dataset, boosting training script, prediction helper, saved model artifact, and evaluation reports.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run The Unified App

```bash
streamlit run app.py
```

Default model paths:

- XSS: `xss/saved_models`
- SQL Injection: `sqli/ml_project/artifacts/sqli_logreg_model.json`
- CSRF: `csrf/artifacts/csrf_boosting_pipeline.joblib`

You can change either path from the app sidebar.

Batch mode accepts manual line-by-line input and upload files. Uploads can be `.csv` or `.xlsx`.

## XSS Module

Run these commands from the `xss/` folder:

```bash
python eda.py
python train.py
python tune.py
python evaluate.py
streamlit run app.py
```

Important files:

- `xss/XSS_dataset.csv`
- `xss/classifier.py`
- `xss/preprocessing.py`
- `xss/train.py`
- `xss/tune.py`
- `xss/evaluate.py`
- `xss/saved_models/best_model.pt`
- `xss/saved_models/tokenizer.json`
- `xss/saved_models/hparams.json`

## SQL Injection Module

Run these commands from the `sqli/ml_project/` folder:

```bash
python train.py
python predict.py "a' or 1 = 1; --"
streamlit run app.py
```

Important files:

- `sqli/ml_project/dataset/Modified_SQL_Dataset.csv`
- `sqli/ml_project/sql_injection_lr/`
- `sqli/ml_project/train.py`
- `sqli/ml_project/predict.py`
- `sqli/ml_project/artifacts/sqli_logreg_model.json`
- `sqli/ml_project/reports/evaluation.json`

## CSRF Module

Run these commands from the project root:

```bash
python csrf/train.py
python csrf/predict.py
streamlit run app.py
```

Important files:

- `csrf/dataset/mitch-master/dataset/features_matrix.csv`
- `csrf/train.py`
- `csrf/predict.py`
- `csrf/artifacts/csrf_boosting_pipeline.joblib`
- `csrf/artifacts/best_params.json`
- `csrf/reports/evaluation.json`

## Batch Test Files

Ready-to-upload test batches are available in `test_batches/` as both `.csv` and `.xlsx` files:

- `sqli_batch_test`
- `xss_batch_test`
- `csrf_batch_test`

## Notes

The unified app preserves the same dark Streamlit UI styling used by the separate module apps. Existing module-level apps remain available for testing each detector independently.
