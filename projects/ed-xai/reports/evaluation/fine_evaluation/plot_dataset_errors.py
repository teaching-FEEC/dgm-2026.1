import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY_JSON = RESULTS_DIR / "errors_summary.json"
OUTPUT_PLOT = RESULTS_DIR / "dataset_error_rates.png"

with open(SUMMARY_JSON) as f:
    data = json.load(f)

CATEGORIES = ("false_positive", "false_negative", "unrecognized")

def get_dataset(image_path: str) -> str:
    return image_path.split("/")[0]


model_dataset_counts: dict[str, dict[str, int]] = {}
all_datasets: set[str] = set()

for model, model_data in data.items():
    counts: dict[str, int] = defaultdict(int)
    for category in CATEGORIES:
        for img in model_data["evidence"][category]:
            ds = get_dataset(img)
            counts[ds] += 1
            all_datasets.add(ds)
    model_dataset_counts[model] = counts

datasets = sorted(all_datasets)
models = list(data.keys())

model_total_errors = {m: data[m]["errors"] for m in models}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Dataset Distribution in Model Errors", fontsize=14, fontweight="bold")

COLORS = plt.cm.Set2.colors  # type: ignore[attr-defined]
x = np.arange(len(models))
bar_width = 0.5

# — Left: stacked % of errors —
bottoms = np.zeros(len(models))
for i, ds in enumerate(datasets):
    pct = np.array([
        model_dataset_counts[m].get(ds, 0) / model_total_errors[m] * 100
        if model_total_errors[m] else 0
        for m in models
    ])
    bars = ax1.bar(x, pct, bar_width, bottom=bottoms, label=ds, color=COLORS[i])
    for j, (p, b) in enumerate(zip(pct, bottoms)):
        if p >= 2:
            ax1.text(x[j], b + p / 2, f"{p:.0f}%", ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
    bottoms += pct

ax1.set_title("Share of Errors per Dataset (%)")
ax1.set_ylabel("% of total errors")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha="right")
ax1.set_ylim(0, 105)
ax1.legend(title="Dataset", bbox_to_anchor=(1, 1), loc="upper left")

# — Right: absolute error counts grouped by dataset —
group_width = 0.8
bar_w = group_width / len(datasets)
for i, ds in enumerate(datasets):
    counts = [model_dataset_counts[m].get(ds, 0) for m in models]
    offset = (i - len(datasets) / 2 + 0.5) * bar_w
    bars = ax2.bar(x + offset, counts, bar_w, label=ds, color=COLORS[i])
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                     str(int(h)), ha="center", va="bottom", fontsize=7)

ax2.set_title("Absolute Error Count per Dataset")
ax2.set_ylabel("Number of errors")
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15, ha="right")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(title="Dataset", bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Saved to {OUTPUT_PLOT}")
