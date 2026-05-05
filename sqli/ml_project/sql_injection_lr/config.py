"""Configuration objects for the SQL injection detection pipeline."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_df_ratio: float = 0.95
    max_features: int = 1800

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.25
    iterations: int = 900
    l2_lambda: float = 0.05
    class_weight: str = "balanced"
    tolerance: float = 1e-7

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.20
    validation_size: float = 0.20
    seed: int = 42
    threshold_beta: float = 2.0


def default_model_grid() -> list[ModelConfig]:
    """Small hyperparameter grid for the from-scratch model."""
    return [
        ModelConfig(learning_rate=0.15, iterations=250, l2_lambda=0.05),
        ModelConfig(learning_rate=0.20, iterations=250, l2_lambda=0.10),
        ModelConfig(learning_rate=0.25, iterations=250, l2_lambda=0.10),
        ModelConfig(learning_rate=0.20, iterations=450, l2_lambda=0.10),
    ]


def default_threshold_grid() -> list[float]:
    """Security-oriented threshold grid with extra focus below 0.50."""
    return [round(value / 100, 2) for value in range(15, 66, 2)]
