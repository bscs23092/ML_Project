"""Dataset loading helpers."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def load_dataset(
    csv_path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
) -> tuple[list[str], list[int]]:
    """Load the SQLi CSV dataset as raw texts and integer labels."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset is empty: {path}")

        available_columns = {name.strip().lower(): name for name in reader.fieldnames}
        text_candidates = [text_column] if text_column else ["sentence", "query", "text", "sentence_text"]
        label_candidates = [label_column] if label_column else ["label", "class", "target"]

        resolved_text_column = next(
            (available_columns[candidate.lower()] for candidate in text_candidates if candidate and candidate.lower() in available_columns),
            None,
        )
        resolved_label_column = next(
            (available_columns[candidate.lower()] for candidate in label_candidates if candidate and candidate.lower() in available_columns),
            None,
        )

        if resolved_text_column is None or resolved_label_column is None:
            missing: list[str] = []
            if resolved_text_column is None:
                missing.append("text")
            if resolved_label_column is None:
                missing.append("label")
            raise ValueError(
                f"Missing required {', '.join(missing)} column(s). "
                f"Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            text = (row.get(resolved_text_column) or "").strip()
            label_raw = (row.get(resolved_label_column) or "").strip()
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
