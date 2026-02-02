import matplotlib.pyplot as plt

# ==============================
# Chinese paper font: SimSun
# ==============================
plt.rcParams["font.family"] = "SimSun"   # 宋体
plt.rcParams["axes.unicode_minus"] = False

# 关键：防止 pdf/svg 里变成 Type 3 字体
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Optional: 统一字号（推荐）
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


def plot_dataset_splits(dataset_name: str, splits: dict, save_dir: str = "."):
    """
    Plot 3 subplots (Train/Val/Test) for a dataset.
    - MOSI/MOSEI: 7-class
    - SIMS: 5-class (no padding)
    Logic is driven by the provided category list in `splits`.
    """

    # Expect splits like:
    # splits = {
    #   "Train": (categories_list, values_list),
    #   "Val":   (categories_list, values_list),
    #   "Test":  (categories_list, values_list),
    # }
    split_order = ["Train", "Val", "Test"]
    split_titles = ["训练集", "验证集", "测试集"]

    # Basic validation & determine categories per split
    for s in split_order:
        if s not in splits:
            raise ValueError(f"{dataset_name}: missing split '{s}' in splits dict.")
        cats, vals = splits[s]
        if len(cats) != len(vals):
            raise ValueError(f"{dataset_name}-{s}: categories and values length mismatch.")

    # Determine a suitable y-limit from all splits
    max_val = max(max(vals) for _, vals in splits.values())
    y_max = max_val * 1.18

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=200, sharey=True)

    for ax, split_name, split_title in zip(axes, split_order, split_titles):
        categories, values = splits[split_name]
        x = list(range(len(categories)))

        bars = ax.bar(x, values, width=0.6)

        ax.set_title(split_title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontfamily='Times New Roman')
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

        # Value labels
        for i, v in enumerate(values):
            ax.text(i, v + y_max * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=11)

        # For readability when many classes (7-class)
        ax.tick_params(axis="x", labelrotation=0)

    axes[0].set_ylabel("百分比 (%)")
    fig.tight_layout()

    # Save
    fig.savefig(f"{save_dir}/{dataset_name.lower()}_distribution_train_val_test.png", bbox_inches="tight")

    plt.show()


# ----------------------------
# Data (percentages) from your figures
# ----------------------------

# 7-class labels for MOSI/MOSEI
cats_7 = [
    "Strongly\nNegative",
    "Weakly\nNegative",
    "Negative",
    "Neutral",
    "Positive",
    "Weakly\nPositive",
    "Strongly\nPositive",
]

# 5-class labels for SIMS
cats_5 = [
    "Weakly\nNegative",
    "Negative",
    "Neutral",
    "Positive",
    "Weakly\nPositive",
]

mosi_splits = {
    "Train": (cats_7, [1.9, 15.5, 17.7, 19.3, 17.7, 24.4, 3.6]),
    "Val":   (cats_7, [4.4, 13.1, 14.0, 21.4, 17.0, 21.8, 8.3]),
    "Test":  (cats_7, [6.7, 22.7, 21.1, 15.5, 16.5, 14.6, 2.9]),
}

mosei_splits = {
    "Train": (cats_7, [1.9, 8.0, 12.5, 41.6, 25.2, 9.5, 1.3]),
    "Val":   (cats_7, [1.3, 5.9, 13.0, 44.6, 25.1, 9.0, 1.0]),
    "Test":  (cats_7, [1.5, 8.6, 11.9, 41.4, 26.4, 9.4, 0.8]),
}

# SIMS (5-class only, no padding)
sims_splits = {
    "Train": (cats_5, [33.0, 21.2, 15.1, 15.2, 15.4]),
    "Val":   (cats_5, [33.1, 21.3, 15.1, 15.1, 15.4]),
    "Test":  (cats_5, [33.0, 21.2, 15.1, 15.1, 15.5]),
}

# ----------------------------
# Run (each dataset produces one figure with 3 subplots)
# ----------------------------
plot_dataset_splits("MOSI", mosi_splits, save_dir=".")
plot_dataset_splits("MOSEI", mosei_splits, save_dir=".")
plot_dataset_splits("SIMS", sims_splits, save_dir=".")
