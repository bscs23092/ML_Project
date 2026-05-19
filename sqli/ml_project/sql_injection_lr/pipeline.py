from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig
from .features import CombinedFeatureExtractor
from .model import LogisticRegressionScratch


class SQLInjectionPipeline:
    def __init__(
        self,
        feature_extractor: CombinedFeatureExtractor | None = None,
        model: LogisticRegressionScratch | None = None,
        threshold: float = 0.5,
    ):
        self.feature_extractor = feature_extractor or CombinedFeatureExtractor()
        self.model = model or LogisticRegressionScratch()
        self.threshold = threshold

    def fit(self, texts: list[str], labels: list[int], model_config: ModelConfig | None = None) -> "SQLInjectionPipeline":
        if model_config is not None:
            self.model = LogisticRegressionScratch.from_config(model_config)
        X = self.feature_extractor.fit_transform(texts)
        self.model.fit(X, labels, n_features=self.feature_extractor.n_features_)
        return self

    def predict_proba(self, texts: list[str]) -> list[float]:
        X = self.feature_extractor.transform(texts)
        return self.model.predict_proba(X)

    def predict(self, texts: list[str], threshold: float | None = None) -> list[int]:
        threshold = self.threshold if threshold is None else threshold
        X = self.feature_extractor.transform(texts)
        return self.model.predict(X, threshold=threshold)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "threshold": self.threshold,
            "feature_extractor": self.feature_extractor.to_dict(),
            "model": self.model.to_dict(),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SQLInjectionPipeline":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            feature_extractor=CombinedFeatureExtractor.from_dict(payload["feature_extractor"]),
            model=LogisticRegressionScratch.from_dict(payload["model"]),
            threshold=float(payload["threshold"]),
        )
