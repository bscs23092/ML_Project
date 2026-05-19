from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def load_dataset(
    csv_path: str | Path,
    text_column: str = "Sentence",
    label_column: str = "Label",
) -> tuple[list[str], list[int]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset is empty: {path}")
        missing = {text_column, label_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required column(s) {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            label_raw = (row.get(label_column) or "").strip()
            if not text:
                continue
            try:
                label = int(label_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label on row {row_number}: {label_raw!r}"
                ) from exc
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 on row {row_number}: {label}")
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"No usable rows found in {path}")
    return texts, labels


def label_distribution(labels: list[int]) -> dict[int, int]:
    counts = Counter(labels)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}
