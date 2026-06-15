from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "mitch-master" / "dataset" / "features_matrix.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

DROP_COLUMNS = ["reqId", "flag"]
LOG1P_COLUMNS = ["numOfParams", "numOfBools", "numOfIds", "numOfBlobs", "reqLen"]
POSITIVE_FLAG = "y"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a boosting model for CSRF-relevant request detection.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to Mitch features_matrix.csv.")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR, help="Directory for model artifacts.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="Directory for reports.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out stratified test fraction.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds for hyperparameter tuning.")
    parser.add_argument("--n-iter", type=int, default=50, help="RandomizedSearchCV iterations.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for CV and permutation importance.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    required = {"reqId", "flag", *LOG1P_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.isna().any().any():
        missing_counts = df.isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing_counts[missing_counts > 0].to_dict()}")

    original_flags = df["flag"].astype(str)
    y = (original_flags == POSITIVE_FLAG).astype(int)
    X = df.drop(columns=DROP_COLUMNS)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Expected all feature columns to be numeric, found: {non_numeric}")

    return X, y, original_flags


def build_pipeline(seed: int) -> Pipeline:
    log_columns = [col for col in LOG1P_COLUMNS]
    passthrough_columns = "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False),
                log_columns,
            )
        ],
        remainder=passthrough_columns,
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("variance", VarianceThreshold()),
            ("clf", classifier),
        ]
    )


def parameter_space() -> dict[str, Any]:
    return {
        "clf__learning_rate": loguniform(0.015, 0.20),
        "clf__max_iter": [100, 150, 200, 300, 450, 600],
        "clf__max_leaf_nodes": [7, 15, 31, 63],
        "clf__max_depth": [2, 3, 4, 5, 7, None],
        "clf__min_samples_leaf": [5, 10, 20, 30, 50, 80],
        "clf__l2_regularization": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.50, 1.0],
        "clf__max_bins": [64, 128, 255],
        "clf__max_features": [0.60, 0.75, 0.90, 1.0],
    }


def label_distribution(y: pd.Series | np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def flag_distribution(flags: pd.Series | np.ndarray) -> dict[str, int]:
    counts = pd.Series(flags).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
    }


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["not_csrf_relevant", "csrf_relevant"],
            zero_division=0,
            output_dict=True,
        ),
    }


def get_selected_feature_names(model: Pipeline, feature_columns: list[str]) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    variance = model.named_steps["variance"]
    transformed_names = preprocessor.get_feature_names_out(feature_columns)
    return transformed_names[variance.get_support()].tolist()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    X, y, original_flags = load_dataset(args.data)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test, flags_train, flags_test = train_test_split(
        X,
        y,
        original_flags,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        estimator=build_pipeline(args.seed),
        param_distributions=parameter_space(),
        n_iter=args.n_iter,
        scoring={
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        },
        refit="average_precision",
        cv=cv,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_model: Pipeline = search.best_estimator_
    oof_probabilities = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]
    threshold_info = find_best_threshold(y_train.to_numpy(), oof_probabilities)
    best_threshold = threshold_info["threshold"]

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    train_oof_metrics = evaluate_predictions(y_train.to_numpy(), oof_probabilities, best_threshold)
    test_metrics = evaluate_predictions(y_test.to_numpy(), test_probabilities, best_threshold)
    test_metrics_at_050 = evaluate_predictions(y_test.to_numpy(), test_probabilities, 0.50)

    selected_features = get_selected_feature_names(best_model, feature_columns)
    permutation = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=15,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    feature_importance = sorted(
        [
            {
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(
                feature_columns,
                permutation.importances_mean,
                permutation.importances_std,
            )
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )

    artifact = {
        "pipeline": best_model,
        "threshold": best_threshold,
        "feature_columns": feature_columns,
        "selected_feature_names_after_variance": selected_features,
        "target_definition": {"csrf_relevant": "flag == 'y'", "not_csrf_relevant": "flag in ['n', 'u', 'm']"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model_path = args.artifact_dir / "csrf_boosting_pipeline.joblib"
    best_params_path = args.artifact_dir / "best_params.json"
    report_path = args.report_dir / "evaluation.json"
    cv_results_path = args.report_dir / "cv_results.csv"
    feature_importance_path = args.report_dir / "permutation_importance.json"

    joblib.dump(artifact, model_path)
    pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

    best_params_report = {
        "model_type": "sklearn.ensemble.HistGradientBoostingClassifier",
        "boosting_note": "XGBoost is not required; this uses scikit-learn histogram gradient boosting.",
        "best_cv_average_precision": float(search.best_score_),
        "best_params": {key: json_safe(value) for key, value in search.best_params_.items()},
        "oof_threshold_selection": threshold_info,
        "cv": {
            "folds": args.cv_folds,
            "splitter": "StratifiedKFold(shuffle=True)",
            "refit_metric": "average_precision",
            "n_iter": args.n_iter,
        },
        "split": {
            "test_size": args.test_size,
            "strategy": "Stratified train/test split; hyperparameters selected by CV only on training split.",
            "random_state": args.seed,
        },
        "artifacts": {
            "model": str(model_path),
            "report": str(report_path),
            "cv_results": str(cv_results_path),
            "feature_importance": str(feature_importance_path),
        },
    }

    report = {
        "dataset": {
            "path": str(args.data),
            "samples": int(len(X)),
            "features_before_pipeline": int(X.shape[1]),
            "original_flag_distribution": flag_distribution(original_flags),
            "binary_distribution": label_distribution(y),
            "train_binary_distribution": label_distribution(y_train),
            "test_binary_distribution": label_distribution(y_test),
            "train_original_flag_distribution": flag_distribution(flags_train),
            "test_original_flag_distribution": flag_distribution(flags_test),
        },
        "leakage_controls": [
            "reqId and flag are removed before modeling.",
            "The held-out test set is never used for hyperparameter tuning or threshold selection.",
            "Log transforms and variance filtering are inside the sklearn Pipeline, so they are refit inside each CV fold.",
            "All splits are stratified on the binary target.",
        ],
        "best_params": best_params_report,
        "selected_features_after_variance": selected_features,
        "train_oof_metrics": train_oof_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0_50_threshold": test_metrics_at_050,
        "top_permutation_importance": feature_importance[:20],
    }

    with best_params_path.open("w", encoding="utf-8") as handle:
        json.dump(best_params_report, handle, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with feature_importance_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_importance, handle, indent=2)

    print("CSRF boosting training complete")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"Selected threshold from out-of-fold train predictions: {best_threshold:.4f}")
    print(
        "Held-out test: "
        f"PR-AUC={test_metrics['average_precision']:.4f} "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved best params: {best_params_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

