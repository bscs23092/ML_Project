"""Train and evaluate the SQL injection logistic regression pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sql_injection_lr.config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    default_model_grid,
    default_threshold_grid,
)
from sql_injection_lr.data import label_distribution, load_dataset
from sql_injection_lr.features import CombinedFeatureExtractor
from sql_injection_lr.metrics import calculate_metrics, format_metrics
from sql_injection_lr.model import LogisticRegressionScratch
from sql_injection_lr.pipeline import SQLInjectionPipeline
from sql_injection_lr.split import stratified_split_indices, take_by_indices
from sql_injection_lr.tuning import grid_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SQL injection detection from scratch.")
    parser.add_argument(
        "--data",
        default="dataset/Modified_SQL_Dataset.csv",
        help="CSV file with Query/Label or Sentence/Label columns.",
    )
    parser.add_argument("--model-out", default="artifacts/sqli_logreg_model.json", help="Where to save the trained pipeline.")
    parser.add_argument("--report-out", default="reports/evaluation.json", help="Where to save metrics and tuning results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits.")
    parser.add_argument("--max-features", type=int, default=1800, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF terms.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of top coefficients to store in the report.")
    return parser.parse_args()


def subset(values: list, indices: list[int]) -> list:
    return take_by_indices(values, indices)


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig(seed=args.seed)
    feature_config = FeatureConfig(max_features=args.max_features, min_df=args.min_df)

    texts, labels = load_dataset(args.data)
    train_indices, test_indices = stratified_split_indices(
        labels,
        test_size=training_config.test_size,
        seed=training_config.seed,
    )
    train_texts = subset(texts, train_indices)
    train_labels = subset(labels, train_indices)
    test_texts = subset(texts, test_indices)
    test_labels = subset(labels, test_indices)

    inner_train_indices, val_indices = stratified_split_indices(
        train_labels,
        test_size=training_config.validation_size,
        seed=training_config.seed + 1,
    )
    inner_train_texts = subset(train_texts, inner_train_indices)
    inner_train_labels = subset(train_labels, inner_train_indices)
    val_texts = subset(train_texts, val_indices)
    val_labels = subset(train_labels, val_indices)

    tuning_extractor = CombinedFeatureExtractor(feature_config)
    X_inner_train = tuning_extractor.fit_transform(inner_train_texts)
    X_val = tuning_extractor.transform(val_texts)

    best_tuning_model, best_threshold, best_result, tuning_results = grid_search(
        X_inner_train,
        inner_train_labels,
        X_val,
        val_labels,
        n_features=tuning_extractor.n_features_,
        model_configs=default_model_grid(),
        thresholds=default_threshold_grid(),
        beta=training_config.threshold_beta,
    )

    best_model_config = ModelConfig(**best_result["params"])

    final_extractor = CombinedFeatureExtractor(feature_config)
    X_train = final_extractor.fit_transform(train_texts)
    X_test = final_extractor.transform(test_texts)
    final_model = LogisticRegressionScratch.from_config(best_model_config)
    final_model.fit(X_train, train_labels, n_features=final_extractor.n_features_)

    validation_probabilities = best_tuning_model.predict_proba(X_val)
    test_probabilities = final_model.predict_proba(X_test)
    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold=best_threshold,
        beta=training_config.threshold_beta,
    )
    test_metrics_at_default = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold=0.5,
        beta=training_config.threshold_beta,
    )

    pipeline = SQLInjectionPipeline(final_extractor, final_model, threshold=best_threshold)
    pipeline.save(args.model_out)

    top_coefficients = final_model.top_coefficients(final_extractor.feature_names(), top_n=args.top_n)
    report = {
        "dataset": {
            "path": args.data,
            "total_samples": len(texts),
            "label_distribution": label_distribution(labels),
            "train_distribution": label_distribution(train_labels),
            "validation_distribution": label_distribution(val_labels),
            "test_distribution": label_distribution(test_labels),
        },
        "training_config": asdict(training_config),
        "feature_config": feature_config.to_dict(),
        "selected_model": best_result,
        "final_test_metrics": test_metrics,
        "final_test_metrics_at_0_50": test_metrics_at_default,
        "validation_threshold_sweep": [
            calculate_metrics(val_labels, validation_probabilities, threshold, beta=training_config.threshold_beta)
            for threshold in default_threshold_grid()
        ],
        "test_threshold_sweep": [
            calculate_metrics(test_labels, test_probabilities, threshold, beta=training_config.threshold_beta)
            for threshold in default_threshold_grid()
        ],
        "top_tuning_results": tuning_results[:5],
        "top_coefficients": top_coefficients,
        "artifacts": {
            "model": args.model_out,
            "report": args.report_out,
        },
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Dataset:", len(texts), label_distribution(labels))
    print("Selected params:", best_result["params"])
    print("Selected validation:", format_metrics(best_result["validation_metrics"]))
    print("Final test:", format_metrics(test_metrics))
    print("Final test at threshold 0.50:", format_metrics(test_metrics_at_default))
    print(f"Saved model: {args.model_out}")
    print(f"Saved report: {args.report_out}")


if __name__ == "__main__":
    main()
