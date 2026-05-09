from pathlib import Path
import json
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from classifier import XSSClassifier
from preprocessing import CharTokenizer, XSSDataset


BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "saved_models"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_hparams() -> dict:
    defaults = {
        "embed_dim": 128,
        "num_filters": 64,
        "dropout": 0.5,
    }
    path = SAVE_DIR / "hparams.json"
    if not path.exists():
        return defaults

    with open(path) as f:
        data = json.load(f)

    return {**defaults, **data}


def main():
    hparams = load_hparams()
    tok = CharTokenizer.load(str(SAVE_DIR / "tokenizer.json"))

    with open(SAVE_DIR / "test_split.pkl", "rb") as f:
        split = pickle.load(f)
    X_test, y_test = split["X_test"], split["y_test"]

    model = XSSClassifier(tok.vocab_size, hparams["embed_dim"], hparams["num_filters"], hparams["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", map_location=DEVICE))
    model.eval()

    with open(SAVE_DIR / "history.json") as f:
        history = json.load(f)

    test_ds = XSSDataset(X_test, y_test, tok)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            all_logits.append(model(xb).cpu())
            all_labels.append(yb)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))

    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
    valid_mask = recall_arr[:-1] >= 0.95
    valid_f1 = np.where(valid_mask, f1_scores, 0)
    best_idx = int(np.argmax(valid_f1))
    best_thresh = float(thresholds[best_idx])
    best_prec = float(precision_arr[best_idx])
    best_rec = float(recall_arr[best_idx])

    if valid_mask.any():
        print(
            f"[Threshold] F1-optimal with recall>=0.95 = {best_thresh:.3f}  "
            f"→ precision={best_prec:.3f}  recall={best_rec:.3f}  f1={f1_scores[best_idx]:.3f}"
        )
    else:
        print(
            f"[Threshold] no threshold met recall>=0.95; falling back to {best_thresh:.3f}  "
            f"→ precision={best_prec:.3f}  recall={best_rec:.3f}  f1={f1_scores[best_idx]:.3f}"
        )

    preds = (probs >= best_thresh).astype(int)

    print("\n── Classification Report ──")
    print(classification_report(labels, preds, target_names=["Benign", "XSS"], digits=4))

    roc_auc = roc_auc_score(labels, probs)
    print(f"ROC-AUC = {roc_auc:.4f}")

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"  False-negative rate (missed XSS) = {fn / (fn + tp):.3f}")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history["train_loss"]) + 1)
    ax0.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    ax0.plot(epochs, history["val_loss"], label="Val loss", color="tomato")
    ax0.set_title("Loss over Epochs")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(epochs, history["val_f1"], label="Val F1", color="steelblue")
    ax1.plot(epochs, history["val_recall"], label="Val Recall", color="tomato")
    ax1.set_title("Val F1 & Recall over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(recall_arr, precision_arr, color="purple", lw=2)
    ax2.axvline(best_rec, color="black", linestyle="--", label=f"Chosen threshold={best_thresh:.2f}")
    ax2.set_title("Precision–Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "XSS"])
    disp.plot(ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title(f"Confusion Matrix (thresh={best_thresh:.2f})")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(probs[labels == 0], bins=50, alpha=0.6, color="steelblue", label="Benign", density=True)
    ax4.hist(probs[labels == 1], bins=50, alpha=0.6, color="tomato", label="XSS", density=True)
    ax4.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold={best_thresh:.2f}")
    ax4.set_title("Predicted Probability Distribution")
    ax4.set_xlabel("P(XSS)")
    ax4.set_ylabel("Density")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(thresholds, precision_arr[:-1], label="Precision", color="steelblue")
    ax5.plot(thresholds, recall_arr[:-1], label="Recall", color="tomato")
    ax5.axvline(best_thresh, color="black", linestyle="--", label=f"Chosen={best_thresh:.2f}")
    ax5.set_title("Precision & Recall vs Threshold")
    ax5.set_xlabel("Threshold")
    ax5.set_ylabel("Score")
    ax5.legend()

    plt.suptitle("XSS CNN — Test Set Evaluation", fontsize=14, fontweight="bold")
    out_path = SAVE_DIR / "evaluation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Done] plots saved → {out_path}")


if __name__ == "__main__":
    main()
