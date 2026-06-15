from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "csrf_boosting_pipeline.joblib"
DEFAULT_DATA_PATH = BASE_DIR / "dataset" / "mitch-master" / "dataset" / "features_matrix.csv"
ATTACK_FLAG = "y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the CSRF model against labeled attack and benign examples."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved joblib model artifact.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Labeled Mitch feature matrix used to sample known attacks and benign requests.",
    )
    parser.add_argument("--attacks", type=int, default=5, help="Number of labeled attack rows to test.")
    parser.add_argument("--benign", type=int, default=5, help="Number of labeled benign rows to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling rows.")
    parser.add_argument(
        "--json-file",
        type=Path,
        default=None,
        help="Optional JSON file containing one row or a list of rows to test instead of sampling from the dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if any sample is misclassified.",
    )
    return parser.parse_args()


def load_artifact(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def load_dataset_samples(data_path: Path, attack_count: int, benign_count: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = {"reqId", "flag"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    attacks = df[df["flag"].astype(str) == ATTACK_FLAG]
    benign = df[df["flag"].astype(str) != ATTACK_FLAG]

    if attacks.empty:
        raise ValueError("No attack rows found in the dataset.")
    if benign.empty:
        raise ValueError("No benign rows found in the dataset.")

    sampled_attacks = attacks.sample(n=min(attack_count, len(attacks)), random_state=seed)
    sampled_benign = benign.sample(n=min(benign_count, len(benign)), random_state=seed)
    samples = pd.concat([sampled_attacks, sampled_benign], ignore_index=True)
    return samples.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_json_samples(json_file: Path) -> pd.DataFrame:
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("JSON input must be an object or a list of objects.")

    if not rows:
        raise ValueError("JSON input did not contain any rows.")

    return pd.DataFrame(rows)


def expected_label_from_flag(flag: Any) -> int:
    return 1 if str(flag).lower() == ATTACK_FLAG else 0


def predict_row(artifact: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in row.index]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    features = pd.DataFrame([{column: row[column] for column in feature_columns}])
    probability = float(artifact["pipeline"].predict_proba(features)[:, 1][0])
    threshold = float(artifact["threshold"])
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "label": "csrf_relevant" if prediction == 1 else "not_csrf_relevant",
        "probability": probability,
        "threshold": threshold,
    }


def main() -> int:
    args = parse_args()
    artifact = load_artifact(args.model)

    if args.json_file:
        samples = load_json_samples(args.json_file)
        if "flag" not in samples.columns:
            raise ValueError("JSON samples must include a 'flag' field for expected-label checking.")
    else:
        samples = load_dataset_samples(args.data, args.attacks, args.benign, args.seed)

    rows = []
    mismatches = 0
    attack_total = 0
    attack_hits = 0

    for index, sample in samples.iterrows():
        result = predict_row(artifact, sample)
        expected = expected_label_from_flag(sample["flag"])
        is_attack = expected == 1
        correct = result["prediction"] == expected

        rows.append(
            {
                "row": index + 1,
                "reqId": sample.get("reqId", "n/a"),
                "expected": "csrf_relevant" if expected == 1 else "not_csrf_relevant",
                "predicted": result["label"],
                "probability": round(result["probability"], 6),
                "threshold": round(result["threshold"], 6),
                "correct": correct,
            }
        )

        if is_attack:
            attack_total += 1
            attack_hits += int(correct and result["prediction"] == 1)
        if not correct:
            mismatches += 1

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    accuracy = float((report["correct"] == True).mean()) if not report.empty else 0.0
    attack_recall = float(attack_hits / attack_total) if attack_total else 0.0
    print()
    print(f"samples={len(report)} accuracy={accuracy:.3f} attack_recall={attack_recall:.3f} mismatches={mismatches}")

    if args.strict and mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())