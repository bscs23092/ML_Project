from __future__ import annotations

import argparse
import random

from sql_injection_lr.data import load_dataset
from sql_injection_lr.metrics import calculate_metrics, format_metrics
from sql_injection_lr.pipeline import SQLInjectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SQL injection with a trained model.")
    parser.add_argument("query", nargs="*", help="Raw query or sentence to classify.")
    parser.add_argument("--model", default="artifacts/sqli_logreg_model.json", help="Trained model artifact.")
    parser.add_argument(
        "--data",
        default="dataset/Modified_SQL_Dataset.csv",
        help="Dataset used for batch testing.",
    )
    parser.add_argument(
        "--batch-test",
        action="store_true",
        help="Sample normal and malicious rows from the dataset and evaluate them in one run.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of normal and malicious samples to test in batch mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for batch sampling.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the tuned threshold.")
    return parser.parse_args()


def sample_batch(texts: list[str], labels: list[int], sample_size: int, seed: int) -> tuple[list[str], list[int]]:
    normal_rows = [(text, label) for text, label in zip(texts, labels) if label == 0]
    malicious_rows = [(text, label) for text, label in zip(texts, labels) if label == 1]
    if len(normal_rows) < sample_size or len(malicious_rows) < sample_size:
        raise ValueError(
            f"Dataset must contain at least {sample_size} rows of each class for batch testing. "
            f"Found normal={len(normal_rows)} malicious={len(malicious_rows)}"
        )

    rng = random.Random(seed)
    selected = rng.sample(normal_rows, sample_size) + rng.sample(malicious_rows, sample_size)
    rng.shuffle(selected)
    sampled_texts = [text for text, _label in selected]
    sampled_labels = [label for _text, label in selected]
    return sampled_texts, sampled_labels


def format_batch_summary(
    data_path: str,
    sample_size: int,
    metrics: dict,
    actual_counts: dict[int, int],
    predicted_counts: dict[int, int],
) -> str:
    cm = metrics["confusion_matrix"]
    separator = "=" * 100
    thin_separator = "-" * 100
    
    lines = [
        "",
        "SQL INJECTION DETECTION - BATCH TEST RESULTS",
        separator,
        "",
        "DATASET INFORMATION:",
        f"  File Path      : {data_path}",
        f"  Samples/Class  : {sample_size}",
        f"  Total Samples  : {sample_size * 2}",
        "",
        "CLASS DISTRIBUTION:",
        f"  Actual Normal     : {actual_counts.get(0, 0):3d} samples",
        f"  Actual Malicious  : {actual_counts.get(1, 0):3d} samples",
        "",
        "MODEL PREDICTIONS:",
        f"  Predicted Normal    : {predicted_counts.get(0, 0):3d} samples",
        f"  Predicted Malicious : {predicted_counts.get(1, 0):3d} samples",
        "",
        thin_separator,
        "PERFORMANCE METRICS",
        thin_separator,
        "",
        f"  Threshold           : {metrics['threshold']:.4f}",
        f"  Accuracy            : {metrics['accuracy']:.4f}",
        f"  Precision           : {metrics['precision']:.4f}",
        f"  Recall              : {metrics['recall']:.4f}",
        f"  F1 Score            : {metrics['f1']:.4f}",
        f"  Specificity         : {metrics['specificity']:.4f}",
        "",
        "CONFUSION MATRIX:",
        f"  True Negatives (TN) : {cm['tn']:3d}  |  False Positives (FP) : {cm['fp']:3d}",
        f"  False Negatives (FN): {cm['fn']:3d}  |  True Positives (TP)  : {cm['tp']:3d}",
        "",
        separator,
    ]
    return "\n".join(lines)


def format_sample_predictions(batch_texts: list[str], batch_labels: list[int], predictions: list[int], probabilities: list[float]) -> str:
    separator = "=" * 120
    thin_separator = "-" * 120
    
    header = f"{'NO.':<5} {'STATUS':<8} {'ACTUAL':<12} {'PREDICTED':<12} {'PROB':<12} QUERY TEXT"
    
    lines = [
        "",
        "SAMPLE PREDICTIONS DETAIL",
        separator,
        header,
        thin_separator,
    ]
    
    for index, (text, actual, predicted, probability) in enumerate(
        zip(batch_texts, batch_labels, predictions, probabilities),
        start=1,
    ):
        actual_label = "MALICIOUS" if actual == 1 else "NORMAL"
        predicted_label = "MALICIOUS" if predicted == 1 else "NORMAL"
        status = "✓" if actual == predicted else "✗"
        
        display_text = text[:80] + "..." if len(text) > 80 else text
        
        lines.append(
            f"{index:03d}   {status}      {actual_label:<12} {predicted_label:<12} {probability:.4f}       {display_text}"
        )
    
    lines.append(separator)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    pipeline = SQLInjectionPipeline.load(args.model)
    threshold = pipeline.threshold if args.threshold is None else args.threshold

    if args.batch_test:
        texts, labels = load_dataset(args.data)
        batch_texts, batch_labels = sample_batch(texts, labels, args.sample_size, args.seed)
        probabilities = pipeline.predict_proba(batch_texts)
        metrics = calculate_metrics(batch_labels, probabilities, threshold=threshold)
        predictions = [int(probability >= threshold) for probability in probabilities]
        actual_counts = {0: batch_labels.count(0), 1: batch_labels.count(1)}
        predicted_counts = {0: predictions.count(0), 1: predictions.count(1)}

        print(format_batch_summary(args.data, args.sample_size, metrics, actual_counts, predicted_counts))
        print(format_sample_predictions(batch_texts, batch_labels, predictions, probabilities))
        return

    query = " ".join(args.query).strip()
    if not query:
        query = input("Enter query/text: ").strip()

    probability = pipeline.predict_proba([query])[0]
    prediction = int(probability >= threshold)
    label = "SQL injection" if prediction == 1 else "Normal"

    separator = "=" * 60
    print(separator)
    print(f"PREDICTION: {prediction} ({label})")
    print(f"PROBABILITY: {probability:.4f}")
    print(f"THRESHOLD: {threshold:.2f}")
    print(separator)


if __name__ == "__main__":
    main()