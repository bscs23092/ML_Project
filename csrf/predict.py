from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "csrf_boosting_pipeline.joblib"
DEFAULT_DATA_PATH = BASE_DIR / "dataset" / "mitch-master" / "dataset" / "features_matrix.csv"


def load_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_from_features(features: dict[str, Any], model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    artifact = load_artifact(model_path)
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([{column: features[column] for column in feature_columns}])
    probability = float(artifact["pipeline"].predict_proba(row)[:, 1][0])
    threshold = float(artifact["threshold"])
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "label": "csrf_relevant" if prediction == 1 else "not_csrf_relevant",
        "probability": probability,
        "threshold": threshold,
    }


def load_demo_features(data_path: Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    demo_row = pd.read_csv(data_path, nrows=1)
    if demo_row.empty:
        raise ValueError(f"Demo dataset is empty: {data_path}")

    features = demo_row.drop(columns=[column for column in ("reqId", "flag") if column in demo_row.columns])
    return features.iloc[0].to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CSRF relevance from a Mitch feature row.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved joblib model artifact.")
    parser.add_argument("--json", type=str, default=None, help="Inline JSON object containing one feature row.")
    parser.add_argument("--json-file", type=Path, default=None, help="Path to a JSON file containing one feature row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_file:
        features = json.loads(args.json_file.read_text(encoding="utf-8"))
    elif args.json:
        features = json.loads(args.json)
    else:
        features = load_demo_features()
        print("No JSON input provided; using the first bundled CSRF dataset row as a demo example.")

    result = predict_from_features(features, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

