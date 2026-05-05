"""From-scratch class-weighted logistic regression with L2 regularization."""

from __future__ import annotations

import math
from collections import Counter

from .config import ModelConfig


SparseRow = tuple[tuple[int, ...], tuple[float, ...]]


class LogisticRegressionScratch:
    """Binary logistic regression trained with full-batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        iterations: int = 900,
        l2_lambda: float = 0.05,
        class_weight: str | dict[int, float] | None = "balanced",
        tolerance: float = 1e-7,
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight
        self.tolerance = tolerance
        self.weights_: list[float] = []
        self.bias_: float = 0.0
        self.n_features_: int = 0
        self.loss_history_: list[float] = []
        self.class_weights_: dict[int, float] = {0: 1.0, 1: 1.0}

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LogisticRegressionScratch":
        return cls(
            learning_rate=config.learning_rate,
            iterations=config.iterations,
            l2_lambda=config.l2_lambda,
            class_weight=config.class_weight,
            tolerance=config.tolerance,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)

    @staticmethod
    def _prepare_rows(X: list[dict[int, float]]) -> list[SparseRow]:
        return [(tuple(row.keys()), tuple(row.values())) for row in X]

    def _resolve_class_weights(self, y: list[int]) -> dict[int, float]:
        if self.class_weight is None:
            return {0: 1.0, 1: 1.0}
        if isinstance(self.class_weight, dict):
            return {
                0: float(self.class_weight.get(0, 1.0)),
                1: float(self.class_weight.get(1, 1.0)),
            }
        if self.class_weight == "balanced":
            counts = Counter(y)
            total = len(y)
            return {
                0: total / (2.0 * max(1, counts.get(0, 0))),
                1: total / (2.0 * max(1, counts.get(1, 0))),
            }
        raise ValueError("class_weight must be None, 'balanced', or a {class: weight} dict")

    def _linear_score(self, row: SparseRow) -> float:
        indices, values = row
        score = self.bias_
        for index, value in zip(indices, values):
            score += self.weights_[index] * value
        return score

    def fit(self, X: list[dict[int, float]], y: list[int], n_features: int) -> "LogisticRegressionScratch":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("Cannot train on an empty dataset")

        rows = self._prepare_rows(X)
        self.n_features_ = n_features
        self.weights_ = [0.0] * n_features
        self.bias_ = 0.0
        self.loss_history_ = []
        self.class_weights_ = self._resolve_class_weights(y)
        sample_weights = [self.class_weights_[label] for label in y]
        weight_denominator = max(1e-12, sum(sample_weights))
        n_samples = len(y)
        previous_loss: float | None = None

        for _iteration in range(self.iterations):
            gradients = [0.0] * n_features
            bias_gradient = 0.0
            weighted_loss = 0.0

            for row, label, sample_weight in zip(rows, y, sample_weights):
                probability = self._sigmoid(self._linear_score(row))
                probability = min(max(probability, 1e-12), 1.0 - 1e-12)
                error = (probability - label) * sample_weight
                bias_gradient += error

                indices, values = row
                for index, value in zip(indices, values):
                    gradients[index] += error * value

                weighted_loss += -sample_weight * (
                    label * math.log(probability)
                    + (1 - label) * math.log(1.0 - probability)
                )

            l2_penalty = sum(weight * weight for weight in self.weights_)
            loss = weighted_loss / weight_denominator
            loss += (self.l2_lambda / (2.0 * n_samples)) * l2_penalty
            self.loss_history_.append(loss)

            for index in range(n_features):
                gradient = gradients[index] / weight_denominator
                gradient += (self.l2_lambda / n_samples) * self.weights_[index]
                self.weights_[index] -= self.learning_rate * gradient
            self.bias_ -= self.learning_rate * (bias_gradient / weight_denominator)

            if previous_loss is not None and abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        return self

    def predict_proba(self, X: list[dict[int, float]]) -> list[float]:
        rows = self._prepare_rows(X)
        return [self._sigmoid(self._linear_score(row)) for row in rows]

    def predict(self, X: list[dict[int, float]], threshold: float = 0.5) -> list[int]:
        return [int(probability >= threshold) for probability in self.predict_proba(X)]

    def top_coefficients(self, feature_names: list[str], top_n: int = 15) -> dict[str, list[tuple[str, float]]]:
        pairs = list(zip(feature_names, self.weights_))
        positive = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]
        negative = sorted(pairs, key=lambda item: item[1])[:top_n]
        return {"positive": positive, "negative": negative}

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_lambda": self.l2_lambda,
            "class_weight": self.class_weight,
            "tolerance": self.tolerance,
            "weights": self.weights_,
            "bias": self.bias_,
            "n_features": self.n_features_,
            "loss_history": self.loss_history_,
            "class_weights": self.class_weights_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LogisticRegressionScratch":
        class_weight = payload["class_weight"]
        if isinstance(class_weight, dict):
            class_weight = {int(key): float(value) for key, value in class_weight.items()}
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            iterations=int(payload["iterations"]),
            l2_lambda=float(payload["l2_lambda"]),
            class_weight=class_weight,
            tolerance=float(payload["tolerance"]),
        )
        model.weights_ = [float(value) for value in payload["weights"]]
        model.bias_ = float(payload["bias"])
        model.n_features_ = int(payload["n_features"])
        model.loss_history_ = [float(value) for value in payload["loss_history"]]
        model.class_weights_ = {int(key): float(value) for key, value in payload["class_weights"].items()}
        return model
