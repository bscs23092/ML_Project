"""Stratified splitting implemented without external libraries."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TypeVar


T = TypeVar("T")


def stratified_split_indices(
    labels: list[int],
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return train and test indices while preserving class ratios."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for class_indices in by_class.values():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        class_test_size = max(1, round(len(shuffled) * test_size))
        test_indices.extend(shuffled[:class_test_size])
        train_indices.extend(shuffled[class_test_size:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def take_by_indices(values: list[T], indices: list[int]) -> list[T]:
    return [values[index] for index in indices]
