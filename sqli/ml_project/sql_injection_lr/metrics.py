from __future__ import annotations


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tn = fp = fn = tp = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 1 and predicted == 1:
            tp += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def calculate_metrics(
    y_true: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
    beta: float = 2.0,
) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    cm = confusion_matrix(y_true, predictions)
    precision = safe_divide(cm["tp"], cm["tp"] + cm["fp"])
    recall = safe_divide(cm["tp"], cm["tp"] + cm["fn"])
    specificity = safe_divide(cm["tn"], cm["tn"] + cm["fp"])
    f1 = fbeta(precision, recall, beta=1.0)
    f_beta = fbeta(precision, recall, beta=beta)
    accuracy = safe_divide(cm["tp"] + cm["tn"], len(y_true))
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        f"f{beta:g}": f_beta,
        "confusion_matrix": cm,
    }


def find_best_threshold(
    y_true: list[int],
    probabilities: list[float],
    thresholds: list[float],
    beta: float = 2.0,
) -> tuple[float, dict]:
    score_key = f"f{beta:g}"
    best_threshold = thresholds[0]
    best_metrics = calculate_metrics(y_true, probabilities, thresholds[0], beta=beta)
    for threshold in thresholds[1:]:
        metrics = calculate_metrics(y_true, probabilities, threshold, beta=beta)
        current_key = (
            metrics[score_key],
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
        )
        best_key = (
            best_metrics[score_key],
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
        )
        if current_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def format_metrics(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    return (
        f"threshold={metrics['threshold']:.2f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"specificity={metrics['specificity']:.4f} "
        f"cm={{tn:{cm['tn']}, fp:{cm['fp']}, fn:{cm['fn']}, tp:{cm['tp']}}}"
    )
