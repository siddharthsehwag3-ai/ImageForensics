from pathlib import Path
import pandas as pd
import matplotlib

# Use a non-interactive backend. No Tkinter required.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "experiments" / "robustness_results.csv"
OUTPUT_DIR = ROOT / "experiments"
FIGURE_DIR = ROOT / "research_figures"

FIGURE_DIR.mkdir(exist_ok=True)

if not RESULTS_FILE.exists():
    raise SystemExit(
        "robustness_results.csv not found. Run robustness_test.py first."
    )

df = pd.read_csv(RESULTS_FILE)

metrics = [
    "precision",
    "recall",
    "f1_score",
    "iou",
    "pixel_accuracy"
]

# Check that the required columns exist.
missing = [column for column in metrics + ["condition"] if column not in df.columns]

if missing:
    raise SystemExit(f"Missing columns in results CSV: {missing}")

# Convert metric columns to numeric values.
for column in metrics:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=metrics)

# Calculate average results for each transformation condition.
summary = (
    df.groupby("condition")[metrics]
    .mean()
    .reset_index()
    .round(5)
)

summary.to_csv(
    OUTPUT_DIR / "robustness_summary.csv",
    index=False
)

print("\n========== AVERAGE RESULTS ==========\n")
print(summary.to_string(index=False))

# Chart 1: Precision, recall, F1 and IoU.
conditions = summary["condition"].tolist()
x = np.arange(len(conditions))
width = 0.19

fig, ax = plt.subplots(figsize=(13, 6))

for i, metric in enumerate(["precision", "recall", "f1_score", "iou"]):
    values = summary[metric].to_numpy()
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=metric.replace("_", " ").title()
    )

ax.set_title("Robustness Evaluation of the Pixel-Difference Baseline")
ax.set_xlabel("Image transformation condition")
ax.set_ylabel("Mean metric score")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=25, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "robustness_metrics.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close(fig)

# Chart 2: F1 score comparison.
fig, ax = plt.subplots(figsize=(11, 5))

values = summary["f1_score"].to_numpy()

bars = ax.bar(
    conditions,
    values,
    width=0.65
)

ax.set_title("F1 Score Under Compression and Resizing")
ax.set_xlabel("Transformation condition")
ax.set_ylabel("Mean F1 score")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "f1_robustness.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close(fig)

# Chart 3: Pixel accuracy comparison.
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    conditions,
    summary["pixel_accuracy"],
    marker="o",
    linewidth=2
)

ax.set_title("Pixel Accuracy Across Transformation Conditions")
ax.set_xlabel("Transformation condition")
ax.set_ylabel("Mean pixel accuracy")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=25)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "pixel_accuracy.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close(fig)

print("\n====================================")
print("RESEARCH CHARTS CREATED SUCCESSFULLY")
print("====================================")
print("Summary CSV:", OUTPUT_DIR / "robustness_summary.csv")
print("Chart 1:", FIGURE_DIR / "robustness_metrics.png")
print("Chart 2:", FIGURE_DIR / "f1_robustness.png")
print("Chart 3:", FIGURE_DIR / "pixel_accuracy.png")
print("Total evaluation rows:", len(df))
