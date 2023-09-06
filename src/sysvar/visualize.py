import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def create_single_figure():
    return plt.subplots(figsize=(8, 5), dpi=800)


def create_double_figure():
    return plt.subplots(1, 2, figsize=(16, 4.5), dpi=800)


def plot_matrix_on_axis(ax, matrix, tick_labels, title):

    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax)

    ax.set_xticks(np.arange(len(tick_labels)) + 0.5, tick_labels)
    ax.set_yticks(np.arange(len(tick_labels)) + 0.5, tick_labels)

    ax.set_xlabel("correction bins")
    ax.set_ylabel("correction bins")

    ax.set_title(title)


def plot_variation_on_axis(ax, x, variation, label, color, linestyle):

    ax.stairs(
        variation,
        x,
        label=label,
        color=color,
        linestyle=linestyle,
    )
