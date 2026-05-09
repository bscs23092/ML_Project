from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classifier import XSSClassifier, count_parameters
from preprocessing import CharTokenizer, XSSDataset, assert_no_leakage


BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "XSS_dataset.csv"
SAVE_DIR = BASE_DIR / "saved_models"
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 4
THRESHOLD = 0.5
SEED = 42

MAX_LEN_CHOICES = [128, 256, 512]
EMBED_DIM_CHOICES = [32, 64, 128]
NUM_FILTERS_CHOICES = [64, 128, 256]
LR_CHOICES = [1e-4, 5e-4, 1e-3]
POS_WEIGHT_CHOICES = [0.40, 0.48, 0.60, 0.86]
DROPOUT_CHOICES = [0.3, 0.5]


@dataclass(frozen=True)
class TrainConfig:
    max_len: int = 512
    embed_dim: int = 128
    num_filters: int = 64
    lr: float = 1e-3
    pos_weight: float = 0.4
    dropout: float = 0.5
    save_dir: str = str(SAVE_DIR)
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    threshold: float = THRESHOLD
    seed: int = SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the XSS CNN model.")
    parser.add_argument("--max-len", type=int, default=512, choices=MAX_LEN_CHOICES)
    parser.add_argument("--embed-dim", type=int, default=128, choices=EMBED_DIM_CHOICES)
    parser.add_argument("--num-filters", type=int, default=64, choices=NUM_FILTERS_CHOICES)
    parser.add_argument("--lr", type=float, default=1e-3, choices=LR_CHOICES)
    parser.add_argument("--pos-weight", type=float, default=0.4, choices=POS_WEIGHT_CHOICES)
    parser.add_argument("--dropout", type=float, default=0.5, choices=DROPOUT_CHOICES)
    parser.add_argument("--save-dir", type=str, default=str(SAVE_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={col: col.strip().lower() for col in df.columns})

    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])

    if "sentence" not in df.columns or "label" not in df.columns:
        if len(df.columns) < 2:
            raise KeyError(f"Required columns not found. Columns: {df.columns.tolist()}")
        df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})

    if "sentence" not in df.columns or "label" not in df.columns:
        raise KeyError(f"Required columns not found. Columns: {df.columns.tolist()}")

    df["sentence"] = df["sentence"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def compute_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    preds_bin = (torch.sigmoid(logits) >= threshold).long().cpu().numpy()
    labels_np = labels.long().cpu().numpy()

    tp = ((preds_bin == 1) & (labels_np == 1)).sum()
    fp = ((preds_bin == 1) & (labels_np == 0)).sum()
    fn = ((preds_bin == 0) & (labels_np == 1)).sum()

    prec = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * recall / (prec + recall + 1e-8)
    return float(f1), float(recall)


def run_training(config: TrainConfig, trial: optuna.Trial | None = None):
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    df = load_dataset(CSV_PATH)
    X = df["sentence"].values
    y = df["label"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=config.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=config.seed)

    print(f"[Split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"[Split] class train={np.bincount(y_train)} val={np.bincount(y_val)} test={np.bincount(y_test)}")
    print(
        f"[Config] max_len={config.max_len} embed_dim={config.embed_dim} num_filters={config.num_filters} "
        f"dropout={config.dropout} lr={config.lr} pos_weight={config.pos_weight} save_dir={save_dir} epochs={config.epochs}"
    )

    tok = CharTokenizer(max_len=config.max_len)
    tok.fit(X_train)
    assert_no_leakage(X_train, X_val, X_test, tok)
    tok.save(str(save_dir / "tokenizer.json"))

    train_ds = XSSDataset(X_train, y_train, tok)
    val_ds = XSSDataset(X_val, y_val, tok)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = XSSClassifier(vocab_size=tok.vocab_size, embed_dim=config.embed_dim, num_filters=config.num_filters, dropout=config.dropout).to(device)
    count_parameters(model)

    pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_recall": []}
    best_val_f1 = -1.0
    patience_ctr = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)

        avg_train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss_total = 0.0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_total += criterion(logits, yb).item() * len(yb)
                all_logits.append(logits)
                all_labels.append(yb)

        avg_val_loss = val_loss_total / len(val_ds)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        val_f1, val_recall = compute_f1(all_logits, all_labels, threshold=config.threshold)

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_f1"].append(float(val_f1))
        history["val_recall"].append(float(val_recall))

        scheduler.step(val_f1)

        print(
            f"[Epoch {epoch:02d}] train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_f1={val_f1:.4f} val_recall={val_recall:.4f}"
        )

        if trial is not None:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> saved best model (val_f1={best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"[EarlyStopping] no val_f1 improvement for {config.patience} epochs. Stop.")
                break

    with open(save_dir / "hparams.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(save_dir / "test_split.pkl", "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)

    print("[Done] Training complete. Artifacts saved:")
    print(f"  - {save_dir / 'best_model.pt'}")
    print(f"  - {save_dir / 'tokenizer.json'}")
    print(f"  - {save_dir / 'hparams.json'}")
    print(f"  - {save_dir / 'history.json'}")
    print(f"  - {save_dir / 'test_split.pkl'}")

    return best_val_f1, history


def main(config: TrainConfig | None = None):
    if config is None:
        args = build_parser().parse_args()
        config = TrainConfig(
            max_len=args.max_len,
            embed_dim=args.embed_dim,
            num_filters=args.num_filters,
            lr=args.lr,
            pos_weight=args.pos_weight,
            dropout=args.dropout,
            save_dir=args.save_dir,
            epochs=args.epochs,
        )

    run_training(config)


if __name__ == "__main__":
    main()
