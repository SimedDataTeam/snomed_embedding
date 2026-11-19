import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

data = pd.read_csv("./data/los_prediction/accuracies.csv")

# convert data to JSON format
# "Baseline": {
#     "Word2Vec": XX,
#     "BERT": XX,
#     ..
# },
# "E5": {
#     "EN_Description": XX,
#     "FR_Description": XX,
#     "Label_only": XX,
# },
# ...
data = data.set_index(["Model_Group", "Model_Name"])["Accuracy"].unstack().to_dict(orient="index")


# ---------------------------------------------------------------------------
# Panel (a): baselines, split into text-based vs. graph-based embeddings
# ---------------------------------------------------------------------------
TEXT_BASED = ["Word2Vec", "BERT", "ELMo", "USE"]
baselines = data["Baseline"]
GRAPH_BASED = [m for m in baselines if m not in TEXT_BASED]

baseline_groups = {
    "Text-based embeddings": TEXT_BASED,
    "Graph-based embeddings": GRAPH_BASED,
}
baseline_colors = {
    "Text-based embeddings": "#1f77b4",
    "Graph-based embeddings": "#ff7f0e",
}

# ---------------------------------------------------------------------------
# Panel (b): encoder models (one colour per encoder)
# ---------------------------------------------------------------------------
encoder_data = {g: m for g, m in data.items() if g != "Baseline"}
encoder_colors = {
    "E5": "#2ca02c",
    "GTE": "#d62728",
    "RoBERTa-bi": "#9467bd",
    "SapBERT": "#8c564b",
    "S-GTE": "#e377c2",
    "S-Qwen2-1.5B": "#17becf",
}
pretty = {"EN_Description": "EN desc.", "FR_Description": "FR desc.", "Label_only": "Label only"}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
YLIM = (94, 98)
n_a = len(baselines) + 1  # one gap between the two baseline families
n_b = sum(len(m) for m in encoder_data.values()) + len(encoder_data) - 1

fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(18, 8),
    sharey=True,
    gridspec_kw={"width_ratios": [n_a, n_b], "wspace": 0.04},
)


def draw_bars(ax, groups, colors, label_fn=lambda x: x):
    """groups: {group_name: {model: acc}} -> draws bars with a gap between groups."""
    x = 0
    ticks, labels, spans = [], [], []
    for g, models in groups.items():
        pos = np.arange(x, x + len(models))
        bars = ax.bar(
            pos,
            list(models.values()),
            color=colors[g],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            width=0.8,
        )
        for b, v in zip(bars, models.values()):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
        ticks.extend(pos)
        labels.extend(label_fn(m) for m in models)
        spans.append((pos[0], pos[-1]))
        x += len(models) + 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.set_xlim(-0.7, spans[-1][1] + 0.7)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for (_, end), (start, _) in zip(spans[:-1], spans[1:]):
        ax.axvline((end + start) / 2, color="black", linewidth=1.5, alpha=0.3)
    return spans


# (a) baselines
baseline_plot = {k: {m: baselines[m] for m in v} for k, v in baseline_groups.items()}
draw_bars(ax_a, baseline_plot, baseline_colors)
ax_a.set_ylim(*YLIM)
ax_a.set_ylabel("Binary Accuracy (%)", fontsize=12, fontweight="bold")
ax_a.set_title("(a) Baselines", fontsize=14, fontweight="bold", pad=12)
ax_a.legend(
    handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.5, alpha=0.85, label=k) for k, c in baseline_colors.items()],
    loc="upper left",
    framealpha=0.9,
)

# (b) encoder models
spans_b = draw_bars(ax_b, encoder_data, encoder_colors, label_fn=lambda m: pretty.get(m, m))
ax_b.set_title("(b) Encoder models", fontsize=14, fontweight="bold", pad=12)
ax_b.tick_params(axis="y", left=False)
for (start, end), name in zip(spans_b, encoder_data):
    ax_b.text((start + end) / 2, YLIM[1] - 0.08, name, ha="center", va="top", fontsize=12, fontweight="bold", color=encoder_colors[name])

# fig.text(0.5, 0.02, "Models", ha="center", fontsize=12, fontweight="bold")
fig.suptitle("Binary Accuracies: Length of Stay Prediction (<= or > 7 days)", fontsize=15, fontweight="bold", y=0.98)
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.27, wspace=0.04)
os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/los_prediction_accuracies.png", dpi=300)

print("Figure saved")