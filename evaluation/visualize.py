"""
evaluation/visualize.py

Visualization tools for face verification under optical degradation:
1. Attention-Shift Plot: relative reliance on regional (jawline) path vs degradation severity
2. Accuracy vs Severity Curves: compare Clean vs Degraded models across degradation axes
3. Confusion Matrix: 3-class verification heatmap
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import LOGS_DIR


def plot_attention_shift(results_json: Path, output_png: Path):
    """
    Plots the novel Attention-Shift metric (README §9):
    Tracks how the network's relative reliance on the fine-grained regional path
    decays as optical degradation severity increases.
    """
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    axes = ["blur", "jpeg", "illumination"]
    clean_ratio = data.get("clean", {}).get("attention_shift", {}).get("regional_ratio_same", 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))

    for axis in axes:
        severities = [0]
        ratios = [clean_ratio]
        for sev in range(1, 6):
            k = f"{axis}_sev{sev}"
            if k in data:
                r = data[k]["attention_shift"]["regional_ratio_same"]
                severities.append(sev)
                ratios.append(r)

        ax.plot(severities, ratios, marker="o", linewidth=2.5, label=f"Axis: {axis.capitalize()}")

    ax.set_title("Novel Metric: Regional Path Reliance vs Optical Degradation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Degradation Severity Level (0 = Clean, 5 = Severe)", fontsize=11)
    ax.set_ylabel("Regional Reliance Ratio (Regional Norm / Total)", fontsize=11)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0 (Clean)", "1", "2", "3", "4", "5"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved Attention-Shift plot to: {output_png}")


def plot_accuracy_vs_degradation(
    clean_results_json: Path,
    degraded_results_json: Optional[Path] = None,
    output_png: Path = LOGS_DIR / "accuracy_vs_severity.png",
):
    """
    Plots 3-class verification accuracy vs degradation severity,
    comparing Clean model baseline against Degraded fine-tuned model.
    """
    with open(clean_results_json, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    degraded_data = None
    if degraded_results_json and Path(degraded_results_json).exists():
        with open(degraded_results_json, "r", encoding="utf-8") as f:
            degraded_data = json.load(f)

    axes = ["blur", "jpeg", "illumination"]
    fig, subplots = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, axis in enumerate(axes):
        ax = subplots[i]
        sevs = [0, 1, 2, 3, 4, 5]

        # Clean model performance
        clean_accs = [clean_data.get("clean", {}).get("three_class_metrics", {}).get("accuracy", 0.0) * 100]
        for s in range(1, 6):
            k = f"{axis}_sev{s}"
            clean_accs.append(clean_data.get(k, {}).get("three_class_metrics", {}).get("accuracy", 0.0) * 100)
        ax.plot(sevs, clean_accs, "r--o", label="Baseline (Clean Model)", linewidth=2)

        # Degraded model performance if available
        if degraded_data:
            deg_accs = [degraded_data.get("clean", {}).get("three_class_metrics", {}).get("accuracy", 0.0) * 100]
            for s in range(1, 6):
                k = f"{axis}_sev{s}"
                deg_accs.append(degraded_data.get(k, {}).get("three_class_metrics", {}).get("accuracy", 0.0) * 100)
            ax.plot(sevs, deg_accs, "g-s", label="Proposed (Degradation-Trained)", linewidth=2)

        ax.set_title(f"Axis: {axis.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Severity Level", fontsize=10)
        if i == 0:
            ax.set_ylabel("3-Class Verification Accuracy (%)", fontsize=10)
        ax.set_xticks(sevs)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("Verification Robustness Across Optical Degradation Axes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved Accuracy comparison plot to: {output_png}")


def plot_confusion_matrix(
    results_json: Path,
    output_png: Path,
    condition: str = "clean",
):
    """
    Plots the 3-class verification confusion matrix heatmap for a given condition.
    Classes: Same Person (0), Twin/Sibling Imposter (1), General Imposter (2).
    """
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if condition not in data:
        print(f"Condition '{condition}' not found in {results_json}. Skipping CM plot.")
        return

    cm = np.array(data[condition].get("three_class_metrics", {}).get("confusion_matrix", []))
    if cm.size == 0:
        print(f"No confusion matrix data for '{condition}'. Skipping CM plot.")
        return

    labels = ["Same Person", "Twin/Sibling", "General Imposter"]
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Pairs", rotation=270, labelpad=15)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate numbers in cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            val = cm[r, c]
            color = "white" if val > thresh else "black"
            ax.text(c, r, f"{val:d}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    acc = data[condition].get("three_class_metrics", {}).get("accuracy", 0.0) * 100
    ax.set_title(f"3-Class Verification ({condition.upper()}) | Acc: {acc:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("--results", type=str, default=None, help="Path to evaluation results JSON")
    parser.add_argument("--compare-results", type=str, default=None, help="Optional second results JSON for comparison")
    parser.add_argument("--clean-results", type=str, default=None, help="Path to Clean model evaluation JSON")
    parser.add_argument("--degraded-results", type=str, default=None, help="Path to Degraded model evaluation JSON")
    parser.add_argument("--output-prefix", type=str, default="eval", help="Prefix for output plot files")

    args = parser.parse_args()

    clean_p = Path(args.clean_results) if args.clean_results else (Path(args.results) if args.results else None)
    deg_p = Path(args.degraded_results) if args.degraded_results else (Path(args.compare_results) if args.compare_results else None)

    if not clean_p and not deg_p:
        print("Please provide at least --results or --clean-results.")
        sys.exit(1)

    primary_p = deg_p if (deg_p and deg_p.exists()) else clean_p

    # 1. Attention-Shift Plot
    att_png = LOGS_DIR / f"{args.output_prefix}_attention_shift.png"
    plot_attention_shift(primary_p, att_png)

    # 2. Confusion Matrices for Clean and Severe Blur conditions
    plot_confusion_matrix(primary_p, LOGS_DIR / f"{args.output_prefix}_cm_clean.png", condition="clean")
    plot_confusion_matrix(primary_p, LOGS_DIR / f"{args.output_prefix}_cm_blur_sev4.png", condition="blur_sev4")

    # 3. Accuracy vs Degradation Comparison Plot (if both available)
    if clean_p and deg_p and clean_p.exists() and deg_p.exists():
        acc_png = LOGS_DIR / f"{args.output_prefix}_accuracy_comparison.png"
        plot_accuracy_vs_degradation(clean_p, deg_p, acc_png)
    elif clean_p and clean_p.exists():
        acc_png = LOGS_DIR / f"{args.output_prefix}_accuracy_curves.png"
        plot_accuracy_vs_degradation(clean_p, None, acc_png)

