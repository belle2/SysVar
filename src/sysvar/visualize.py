from typing import Union

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PALETTE = sns.color_palette("colorblind")


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


def plot_variation_on_axis(
    ax: Axes, x: np.ndarray, variation: np.ndarray, index: Union[None, int] = None
):

    """
    Plots a variation on a given axis.
    The absence of a value for the index arguments shows that this is a nominal template.
    The function creates the correct labels, colors and linestyle based on the value
    of the index

    Args:
        ax (Axes): The axis to plot on.
        x (np.ndarray): The x values for the plot.
        variation (np.ndarray): The variation values to plot.
        index (int, None): The index of the variation (None for nominal).

    Returns:
        None

    """

    if index is None:
        label = "Nominal weight"
        color = "black"
        linestyle = "dashed"
    else:
        label = f"variation {index}"
        color = (PALETTE[index],)
        linestyle = "solid"

    ax.stairs(variation, x, label=label, color=color, linestyle=linestyle)
