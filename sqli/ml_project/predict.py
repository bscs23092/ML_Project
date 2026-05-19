from __future__ import annotations

import argparse

from sql_injection_lr.pipeline import SQLInjectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SQL injection with a trained model.")
    parser.add_argument("query", nargs="*", help="Raw query or sentence to classify.")
    parser.add_argument("--model", default="artifacts/sqli_logreg_model.json", help="Trained model artifact.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the tuned threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("Enter query/text: ").strip()

    pipeline = SQLInjectionPipeline.load(args.model)
    threshold = pipeline.threshold if args.threshold is None else args.threshold
    probability = pipeline.predict_proba([query])[0]
    prediction = int(probability >= threshold)
    label = "SQL injection" if prediction == 1 else "Normal"

    print(f"Prediction: {prediction} ({label})")
    print(f"Probability: {probability:.4f}")
    print(f"Threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
