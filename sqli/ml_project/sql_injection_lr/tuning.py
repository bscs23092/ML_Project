"""Simple hyperparameter and threshold tuning."""

from __future__ import annotations

from dataclasses import asdict

from .config import ModelConfig
from .metrics import calculate_metrics, find_best_threshold
from .model import LogisticRegressionScratch


def grid_search(
    X_train: list[dict[int, float]],
    y_train: list[int],
    X_val: list[dict[int, float]],
    y_val: list[int],
    n_features: int,
    model_configs: list[ModelConfig],
    thresholds: list[float],
    beta: float = 2.0,
) -> tuple[LogisticRegressionScratch, float, dict, list[dict]]:
    """Train each config, tune threshold on validation data, and return the best."""
    if not model_configs:
        raise ValueError("model_configs cannot be empty")

    score_key = f"f{beta:g}"
    best_model: LogisticRegressionScratch | None = None
    best_threshold = 0.5
    best_result: dict | None = None
    all_results: list[dict] = []

    for config in model_configs:
        model = LogisticRegressionScratch.from_config(config)
        model.fit(X_train, y_train, n_features=n_features)
        probabilities = model.predict_proba(X_val)
        threshold, threshold_metrics = find_best_threshold(
            y_val,
            probabilities,
            thresholds,
            beta=beta,
        )
        default_metrics = calculate_metrics(y_val, probabilities, threshold=0.5, beta=beta)
        result = {
            "params": asdict(config),
            "best_threshold": threshold,
            "validation_metrics": threshold_metrics,
            "validation_metrics_at_0_50": default_metrics,
            "iterations_run": len(model.loss_history_),
            "final_loss": model.loss_history_[-1] if model.loss_history_ else None,
        }
        all_results.append(result)

        current_key = (
            threshold_metrics[score_key],
            threshold_metrics["f1"],
            threshold_metrics["recall"],
            threshold_metrics["precision"],
        )
        if best_result is None:
            best_key = (-1.0, -1.0, -1.0, -1.0)
        else:
            best_metrics = best_result["validation_metrics"]
            best_key = (
                best_metrics[score_key],
                best_metrics["f1"],
                best_metrics["recall"],
                best_metrics["precision"],
            )
        if current_key > best_key:
            best_model = model
            best_threshold = threshold
            best_result = result

    assert best_model is not None and best_result is not None
    all_results.sort(
        key=lambda result: (
            result["validation_metrics"][score_key],
            result["validation_metrics"]["f1"],
            result["validation_metrics"]["recall"],
            result["validation_metrics"]["precision"],
        ),
        reverse=True,
    )
    return best_model, best_threshold, best_result, all_results
