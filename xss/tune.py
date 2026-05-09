from __future__ import annotations

from pathlib import Path

import json
import optuna

from train import TrainConfig, run_training


BASE_DIR = Path(__file__).parent
SAVE_ROOT = BASE_DIR / "saved_models"
STUDY_DB = SAVE_ROOT / "optuna_study.db"


MAX_LEN_CHOICES = [128, 256, 512]
EMBED_DIM_CHOICES = [32, 64, 128]
NUM_FILTERS_CHOICES = [64, 128, 256]
LR_CHOICES = [1e-4, 5e-4, 1e-3]
POS_WEIGHT_CHOICES = [0.40, 0.48, 0.60, 0.86]
DROPOUT_CHOICES = [0.3, 0.5]


def objective(trial: optuna.Trial) -> float:
    params = {
        "max_len": trial.suggest_categorical("max_len", MAX_LEN_CHOICES),
        "embed_dim": trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES),
        "num_filters": trial.suggest_categorical("num_filters", NUM_FILTERS_CHOICES),
        "lr": trial.suggest_categorical("lr", LR_CHOICES),
        "pos_weight": trial.suggest_categorical("pos_weight", POS_WEIGHT_CHOICES),
        "dropout": trial.suggest_categorical("dropout", DROPOUT_CHOICES),
    }

    trial_dir = SAVE_ROOT / f"trial_{trial.number}"
    config = TrainConfig(
        max_len=params["max_len"],
        embed_dim=params["embed_dim"],
        num_filters=params["num_filters"],
        lr=params["lr"],
        pos_weight=params["pos_weight"],
        dropout=params["dropout"],
        save_dir=str(trial_dir),
        epochs=10,
    )

    best_val_f1, _ = run_training(config, trial=trial)
    return best_val_f1


def main() -> None:
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="xss_cnn_tuning",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=25, gc_after_trial=True)

    print("Best params:", study.best_params)
    print("Best val F1:", study.best_value)

    best_path = SAVE_ROOT / "best_trial.json"
    with open(best_path, "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=2)

    print(f"Saved best study summary to {best_path}")


if __name__ == "__main__":
    main()
