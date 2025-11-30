import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# Set up the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for each group
colors = {
    "Baseline": "#1f77b4",
    "E5": "#ff7f0e",
    "GTE": "#2ca02c",
    "RoBERTa-bi": "#d62728",
    "SapBERT": "#9467bd",
    "S-GTE": "#8c564b",
    "S-Qwen2-1.5B": "#e377c2",
}

# Plot each group
x_position = 0
x_ticks = []
x_labels = []
group_positions = []

for group, models in data.items():
    accuracies = list(models.values())
    model_names = list(models.keys())
    n_models = len(accuracies)

    # Create positions for this group
    positions = np.arange(x_position, x_position + n_models)

    # Plot bars
    ax.bar(positions, accuracies, color=colors[group], label=group, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Store positions for x-axis labels
    x_ticks.extend(positions)
    x_labels.extend(model_names)
    group_positions.append((x_position, x_position + n_models - 1))

    # Update position for next group (with gap)
    x_position += n_models + 1

# Add horizontal reference lines
for y in [94, 95, 96, 97, 98]:
    ax.axhline(y=y, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(-0.5, y, f"{y}%", va="center", ha="right", fontsize=9, color="gray")

# Customize the plot
ax.set_ylabel("Binary Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Models", fontsize=12, fontweight="bold")
ax.set_title("Binary Accuracies: Length of Stay Prediction (<= or > 7 days)", fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
ax.set_ylim(94, 98)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle=":")

# Add vertical separators between groups
for i in range(len(group_positions) - 1):
    sep_pos = (group_positions[i][1] + group_positions[i + 1][0]) / 2 + 0.5
    ax.axvline(x=sep_pos, color="black", linestyle="-", linewidth=1.5, alpha=0.3)

plt.tight_layout()

# Save the figure
plt.savefig("./figures/los_prediction_accuracies.png", dpi=300)
