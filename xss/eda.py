from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(__file__).parent / "XSS_dataset.csv"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={col: col.strip().lower() for col in df.columns})

    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])

    if "sentence" not in df.columns or "label" not in df.columns:
        if len(df.columns) < 2:
            raise ValueError("Expected at least two columns containing sentence and label data")
        df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})

    if "sentence" not in df.columns or "label" not in df.columns:
        raise KeyError(f"Could not find required columns. Available columns: {df.columns.tolist()}")

    df["label"] = df["label"].astype(int)
    return df


def main():
    df = load_data(CSV_PATH)
    print("Columns:", df.columns.tolist())
    print("Shape:  ", df.shape)

    print("\n── Class distribution ──")
    vc = df["label"].value_counts()
    print(vc)
    print(f"  Imbalance ratio  (neg:pos) = {vc[0]}:{vc[1]}  →  pos_weight ≈ {vc[0] / vc[1]:.2f}")

    print("\n── Missing values ──")
    print(df.isnull().sum())

    df["length"] = df["sentence"].astype(str).str.len()
    print("\n── Length stats ──")
    print(df.groupby("label")["length"].describe().T)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    vc.plot(kind="bar", ax=ax0, color=["steelblue", "tomato"], edgecolor="black")
    ax0.set_title("Class Distribution")
    ax0.set_xlabel("Label (0=benign, 1=XSS)")
    ax0.set_ylabel("Count")
    ax0.set_xticklabels(["Benign (0)", "XSS (1)"], rotation=0)
    for p in ax0.patches:
        ax0.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=10)

    ax1 = fig.add_subplot(gs[0, 1])
    for label, color, name in [(0, "steelblue", "Benign"), (1, "tomato", "XSS")]:
        subset = df[df["label"] == label]["length"]
        ax1.hist(subset, bins=50, alpha=0.6, color=color, label=name, density=True)
    ax1.set_title("Sequence Length Distribution")
    ax1.set_xlabel("Character length")
    ax1.set_ylabel("Density")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    for label, color, name in [(0, "steelblue", "Benign"), (1, "tomato", "XSS")]:
        lengths = np.sort(df[df["label"] == label]["length"].values)
        cdf = np.arange(1, len(lengths) + 1) / len(lengths)
        ax2.plot(lengths, cdf, color=color, label=name)
    ax2.axvline(512, color="black", linestyle="--", label="512 cutoff")
    ax2.set_title("CDF of Sequence Lengths")
    ax2.set_xlabel("Character length")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    short_counts = df.groupby("label").apply(lambda g: (g["length"] < 10).sum())
    short_counts.plot(kind="bar", ax=ax3, color=["steelblue", "tomato"], edgecolor="black")
    ax3.set_title("Samples with length < 10")
    ax3.set_xlabel("Label")
    ax3.set_ylabel("Count")
    ax3.set_xticklabels(["Benign (0)", "XSS (1)"], rotation=0)

    plt.suptitle("XSS Dataset — EDA", fontsize=14, fontweight="bold")
    out_path = Path(__file__).parent / "eda_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
