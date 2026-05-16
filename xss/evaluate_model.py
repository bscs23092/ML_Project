from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classifier import XSSClassifier
from preprocessing import CharTokenizer, XSSDataset


BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "saved_models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "evaluation_plots"
DATASET_PATH = BASE_DIR / "XSS_dataset.csv"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


C = dict(
    navy="#16233B",
    slate="#31415E",
    steel="#4F79B6",
    mist="#89A9D6",
    fog="#D7E2F2",
    cream="#F6F8FB",
    white="#FFFFFF",
    alert_red="#C0392B",
    safe_green="#1E7A4A",
    amber="#C57B12",
    charcoal="#26313F",
    mid_grey="#6A7280",
    light_grey="#E6EBF2",
    border="#CAD4E2",
)


def apply_style() -> None:
    rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "axes.edgecolor": C["border"],
            "axes.facecolor": C["cream"],
            "figure.facecolor": C["white"],
            "grid.color": C["light_grey"],
            "grid.linewidth": 0.6,
            "xtick.color": C["mid_grey"],
            "ytick.color": C["mid_grey"],
            "text.color": C["charcoal"],
            "axes.labelcolor": C["charcoal"],
            "axes.titlecolor": C["navy"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.97,
            "legend.edgecolor": C["border"],
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": C["white"],
        }
    )


apply_style()


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    with open(path) as f:
        data = json.load(f)
    if default:
        return {**default, **data}
    return data


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


def compute_lengths(texts: list[str]) -> list[int]:
    return [len(str(text)) for text in texts]


def summarize_threshold_metrics(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    rows = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        specificity = tn / (tn + fp + 1e-8)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": float(specificity),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return {"threshold_sweep": rows}


def load_hparams() -> dict:
    defaults = {"embed_dim": 128, "num_filters": 64, "dropout": 0.5, "threshold": 0.5, "max_len": 512}
    return load_json(SAVE_DIR / "hparams.json", defaults)


def evaluate_model(hparams: dict):
    tokenizer = CharTokenizer.load(str(SAVE_DIR / "tokenizer.json"))
    dataset = load_dataset(DATASET_PATH)
    seed = int(hparams.get("seed", 42))
    x = dataset["sentence"].values
    y = dataset["label"].values

    _, x_tmp, _, y_tmp = train_test_split(x, y, test_size=0.30, stratify=y, random_state=seed)
    _, x_test, _, y_test = train_test_split(x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    y_test = np.asarray(y_test, dtype=int)

    model = XSSClassifier(
        tokenizer.vocab_size,
        hparams["embed_dim"],
        hparams["num_filters"],
        hparams["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", map_location=DEVICE))
    model.eval()

    test_ds = XSSDataset(x_test, y_test, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    with open(SAVE_DIR / "history.json") as f:
        history = json.load(f)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            all_logits.append(model(xb).cpu())
            all_labels.append(yb)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    threshold = float(hparams.get("threshold", 0.5))
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(labels, preds, target_names=["Benign", "XSS"], digits=4, output_dict=True),
    }

    class_counts = dataset["label"].value_counts().sort_index()
    benign_texts = dataset.loc[dataset["label"] == 0, "sentence"].tolist()
    xss_texts = dataset.loc[dataset["label"] == 1, "sentence"].tolist()
    benign_lengths = compute_lengths(benign_texts)
    xss_lengths = compute_lengths(xss_texts)

    summary = {
        "model": {
            "architecture": "CharCNN",
            "embed_dim": int(hparams["embed_dim"]),
            "num_filters": int(hparams["num_filters"]),
            "dropout": float(hparams["dropout"]),
            "max_len": int(hparams.get("max_len", 512)),
            "threshold": threshold,
        },
        "dataset": {
            "total_samples": int(len(dataset)),
            "benign": int(class_counts.get(0, 0)),
            "xss": int(class_counts.get(1, 0)),
            "benign_ratio": float(class_counts.get(0, 0) / max(class_counts.get(1, 1), 1)),
            "avg_benign_length": float(np.mean(benign_lengths)) if benign_lengths else 0.0,
            "avg_xss_length": float(np.mean(xss_lengths)) if xss_lengths else 0.0,
        },
        "evaluation": metrics,
        "history": {
            "epochs": len(history.get("train_loss", [])),
            "best_val_f1": float(max(history.get("val_f1", [0.0]))),
            "final_train_loss": float(history.get("train_loss", [0.0])[-1]),
            "final_val_loss": float(history.get("val_loss", [0.0])[-1]),
        },
    }

    evaluation = {
        "threshold": threshold,
        "predictions": {
            "probs": [float(x) for x in probs],
            "preds": [int(x) for x in preds],
            "labels": [int(x) for x in labels],
        },
    }

    return history, summary, evaluation, labels, probs, preds, x_test, y_test, benign_lengths, xss_lengths


def fig_training(history: dict) -> None:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_f1 = history.get("val_f1", [])
    val_recall = history.get("val_recall", [])

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    ax_loss = fig.add_subplot(gs[0, :])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_gap = fig.add_subplot(gs[1, 1])

    fig.suptitle("XSS CharCNN Training Curves", fontsize=15, fontweight="bold", color=C["navy"], y=1.01)

    ax_loss.plot(epochs, train_loss, color=C["steel"], lw=2.2, marker="o", markersize=4, label="Train loss")
    ax_loss.plot(epochs, val_loss, color=C["alert_red"], lw=2.2, marker="s", markersize=4, label="Validation loss")
    ax_loss.set_title("Loss Curve", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary Cross-Entropy Loss")
    ax_loss.grid(True, alpha=0.4)
    ax_loss.legend(loc="upper right")

    ax_metrics.plot(epochs, val_f1, color=C["safe_green"], lw=2.2, marker="o", markersize=4, label="Validation F1")
    ax_metrics.plot(epochs, val_recall, color=C["amber"], lw=2.2, marker="s", markersize=4, label="Validation Recall")
    ax_metrics.set_title("Training Curve", fontweight="bold")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_ylim(0.0, 1.02)
    ax_metrics.grid(True, alpha=0.4)
    ax_metrics.legend(loc="lower right")

    loss_gap = [abs(t - v) for t, v in zip(train_loss, val_loss)]
    ax_gap.bar(epochs, loss_gap, color=C["mist"], edgecolor=C["white"], width=0.7)
    ax_gap.set_title("Generalisation Gap by Epoch", fontweight="bold")
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("|Train - Val Loss|")
    ax_gap.grid(axis="y", alpha=0.4)

    path = PLOT_DIR / "01_training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_dataset_distribution(benign_lengths: list[int], xss_lengths: list[int], summary: dict) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_len = fig.add_subplot(gs[0, 1:])
    ax_info = fig.add_subplot(gs[1, :])

    fig.suptitle("Dataset Distribution and Input Structure", fontsize=15, fontweight="bold", color=C["navy"], y=1.01)

    counts = [summary["dataset"]["benign"], summary["dataset"]["xss"]]
    classes = ["Benign", "XSS"]
    bars = ax_bar.bar(classes, counts, color=[C["steel"], C["alert_red"]], width=0.55, edgecolor=C["white"], linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02, f"{cnt:,}", ha="center", va="bottom", fontweight="bold")
    ax_bar.set_title("Class Distribution", fontweight="bold")
    ax_bar.set_ylabel("Samples")
    ax_bar.set_ylim(0, max(counts) * 1.18)
    ax_bar.grid(axis="y", alpha=0.45)

    bins = np.arange(0, max(max(benign_lengths or [0]), max(xss_lengths or [0])) + 10, 10)
    if len(bins) < 5:
        bins = np.arange(0, 201, 10)
    ax_len.hist(benign_lengths, bins=bins, color=C["steel"], alpha=0.68, label="Benign", edgecolor=C["white"], lw=0.4)
    ax_len.hist(xss_lengths, bins=bins, color=C["alert_red"], alpha=0.68, label="XSS", edgecolor=C["white"], lw=0.4)
    if benign_lengths:
        ax_len.axvline(float(np.median(benign_lengths)), color=C["navy"], ls="--", lw=1.5, label=f"Median benign = {int(np.median(benign_lengths))}")
    if xss_lengths:
        ax_len.axvline(float(np.median(xss_lengths)), color="#7B1A14", ls="--", lw=1.5, label=f"Median XSS = {int(np.median(xss_lengths))}")
    ax_len.set_title("Character Length Distribution", fontweight="bold")
    ax_len.set_xlabel("Characters per sample")
    ax_len.set_ylabel("Frequency")
    ax_len.grid(axis="y", alpha=0.45)
    ax_len.legend(fontsize=8.5)

    ax_info.axis("off")
    lines = [
        ("Dataset Summary", True, C["navy"]),
        (f"Total samples:     {summary['dataset']['total_samples']:,}", False, C["charcoal"]),
        (f"Benign:            {summary['dataset']['benign']:,}", False, C["charcoal"]),
        (f"XSS:               {summary['dataset']['xss']:,}", False, C["charcoal"]),
        (f"Average benign length: {summary['dataset']['avg_benign_length']:.1f}", False, C["charcoal"]),
        (f"Average XSS length:    {summary['dataset']['avg_xss_length']:.1f}", False, C["charcoal"]),
        ("", False, C["charcoal"]),
        ("Why this matters", True, C["navy"]),
        ("The model is character-level, so sample length and", False, C["charcoal"]),
        ("payload shape are useful diagnostics alongside class balance.", False, C["charcoal"]),
        ("XSS strings often carry dense event handlers, attributes,", False, C["charcoal"]),
        ("and script fragments that create a distinctive length profile.", False, C["charcoal"]),
    ]
    y = 0.96
    for text, bold, color in lines:
        if not text:
            y -= 0.03
            continue
        ax_info.text(0.04, y, text, transform=ax_info.transAxes, va="top", fontsize=8.7, fontweight="bold" if bold else "normal", color=color)
        y -= 0.072 if bold else 0.056

    path = PLOT_DIR / "02_dataset_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_confusion_matrix(labels: np.ndarray, preds: np.ndarray, threshold: float, metrics: dict) -> None:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.35)
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    fig.suptitle(f"Confusion Metrics  —  Operating Threshold = {threshold:.2f}", fontsize=15, fontweight="bold", color=C["navy"], y=1.03)

    cmap = LinearSegmentedColormap.from_list("xss_blue", [C["cream"], C["fog"], C["mist"], C["steel"], C["navy"]])
    ax_cm.imshow(cm, cmap=cmap, aspect="equal", vmin=0, vmax=max(cm.max(), 1))
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted Benign", "Predicted XSS"])
    ax_cm.set_yticklabels(["Actual Benign", "Actual XSS"])
    ax_cm.set_xlabel("Model Prediction")
    ax_cm.set_ylabel("Ground Truth")
    ax_cm.set_title("Confusion Matrix", fontweight="bold", pad=12)

    labels_text = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    desc = [["Benign query allowed", "Benign query flagged"], ["XSS missed", "XSS detected"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            share = val / max(cm.max(), 1)
            fg = "white" if share > 0.55 else C["charcoal"]
            ax_cm.text(j, i - 0.16, f"{val}", ha="center", va="center", fontsize=21, fontweight="bold", color=fg)
            ax_cm.text(j, i + 0.08, f"{100 * val / max(cm.sum(), 1):.1f}%", ha="center", va="center", fontsize=8.5, color=fg)
            ax_cm.text(j, i + 0.28, labels_text[i][j], ha="center", va="center", fontsize=8.2, color=fg, fontweight="bold")
            ax_cm.text(j, i + 0.43, desc[i][j], ha="center", va="center", fontsize=7.5, color=fg if not (i == 1 and j == 0) else ("white" if share > 0.55 else C["alert_red"]))

    ax_cm.axvline(0.5, color="white", lw=2.5)
    ax_cm.axhline(0.5, color="white", lw=2.5)

    ax_info.axis("off")
    rows = [
        ("Evaluation Summary", True, C["navy"]),
        (f"Accuracy:          {metrics['accuracy'] * 100:.2f}%", False, C["charcoal"]),
        (f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%", False, C["charcoal"]),
        (f"Precision:         {metrics['precision'] * 100:.2f}%", False, C["charcoal"]),
        (f"Recall / TPR:      {metrics['recall'] * 100:.2f}%", False, C["charcoal"]),
        (f"Specificity:       {metrics['specificity'] * 100:.2f}%", False, C["charcoal"]),
        (f"F1 score:          {metrics['f1'] * 100:.2f}%", False, C["charcoal"]),
        (f"MCC:               {metrics['mcc']:.4f}", False, C["charcoal"]),
        ("", False, C["charcoal"]),
        ("Security note", True, C["navy"]),
        (f"False negatives: {fn}  |  False positives: {fp}", False, C["charcoal"]),
        ("For XSS detection, missed attacks are the critical risk.", False, C["charcoal"]),
        ("False positives are review overhead, but false negatives", False, C["charcoal"]),
        ("represent a direct exposure path and should be minimised.", False, C["charcoal"]),
    ]

    y = 0.96
    for text, bold, color in rows:
        if not text:
            y -= 0.03
            continue
        ax_info.text(0.03, y, text, transform=ax_info.transAxes, va="top", fontsize=8.8, fontweight="bold" if bold else "normal", color=color)
        y -= 0.073 if bold else 0.057

    path = PLOT_DIR / "03_confusion_metrics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_json_outputs(summary: dict, evaluation: dict) -> None:
    with open(REPORT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(REPORT_DIR / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    print("Generating XSS evaluation reports...")
    hparams = load_hparams()
    history, summary, evaluation, labels, probs, preds, x_test, y_test, benign_lengths, xss_lengths = evaluate_model(hparams)

    save_json_outputs(summary, evaluation)

    fig_training(history)
    fig_dataset_distribution(benign_lengths, xss_lengths, summary)
    fig_confusion_matrix(labels, preds, summary["evaluation"]["threshold"], summary["evaluation"])

    print(f"\nDone. Reports saved to: {REPORT_DIR}")
    for path in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {path.name}")
    print("  evaluation_summary.json")
    print("  evaluation.json")


if __name__ == "__main__":
    main()