import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY_JSON = RESULTS_DIR / "errors_summary.json"
OUTPUT_PLOT = RESULTS_DIR / "error_rates.png"

with open(SUMMARY_JSON) as f:
    data = json.load(f)

models = list(data.keys())
false_positive_pct = [len(data[m]["evidence"]["false_positive"]) / data[m]["total"] * 100 for m in models]
false_negative_pct = [len(data[m]["evidence"]["false_negative"]) / data[m]["total"] * 100 for m in models]
unrecognized_pct  = [len(data[m]["evidence"]["unrecognized"])  / data[m]["total"] * 100 for m in models]

x = np.arange(len(models))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars_fp = ax.bar(x - bar_width, false_positive_pct, bar_width, label="False Positive (real → fake)")
bars_fn = ax.bar(x,             false_negative_pct, bar_width, label="False Negative (fake → real)")
bars_ur = ax.bar(x + bar_width, unrecognized_pct,  bar_width, label="Unrecognized")

for bars in (bars_fp, bars_fn, bars_ur):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.2f}%",
                ha="center", va="bottom", fontsize=8,
            )

ax.set_title("Error Rate by Model and Category")
ax.set_ylabel("Error Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, max(false_positive_pct + false_negative_pct + unrecognized_pct) * 1.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Saved to {OUTPUT_PLOT}")
