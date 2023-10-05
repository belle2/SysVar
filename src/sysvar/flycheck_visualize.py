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


def create_triple_figure():
    return plt.subplots(1, 3, figsize=(16, 4.5), dpi=800)


def plot_matrix_on_axis(ax, matrix, tick_labels, title, axes_labels):

    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax)

    ax.set_xticks(np.arange(len(tick_labels)) + 0.5, tick_labels)
    ax.set_yticks(np.arange(len(tick_labels)) + 0.5, tick_labels)

    ax.set_xlabel(axes_labels)
    ax.set_ylabel(axes_labels)

    ax.set_title(title)


def plot_variation_on_axis(
    ax: Axes,
    x: np.ndarray,
    variation: np.ndarray,
    index: Union[None, int] = None,
    plot_func: str = "step",
):

    """
    Plots a variation on a given axis.
    The absence of a value for the index arguments shows that this is a nominal template.
    The function creates the correct labels, colors and linestyle based on the value
    of the index.

    Args:
        ax (Axes): The axis to plot on.
        x (np.ndarray): The x values for the plot. Should be the bin mid values as the
            method makes use of the matplotlib's steps method
        variation (np.ndarray): The variation values to plot.
        index (int, None): The index of the variation (None for nominal).
        plot_func (str): name of the matplotlib.pyplot function to use for the plot


    Returns:
        None

    """

    if index is None:
        label = "Nominal weight"
        color = "black"
        linestyle = "dashed"
    else:
        label = f"variation {index}"
        color = PALETTE[index]
        linestyle = "solid"

    if plot_func == "step":
        ax.step(x, variation, label=label, color=color, linestyle=linestyle)
    elif plot_func == "stairs":
        ax.stairs(variation, x, label=label, color=color, linestyle=linestyle)
    else:
        raise ValueError(
            f"plot_func argument should be either 'step' or 'stairs' but you passed {plot_func}"
        )


def plot_gaussian_variation_on_axis(
    ax: Axes, mean: float, variations: np.ndarray, string: str
):
    """
    Plot a Gaussian variation on a given axis.

    Args:
        ax (Axes): The axis to plot on.
        mean (float): The mean value.
        variations (np.ndarray): The array of variations.
        string (str): A description string.

    Returns:
        None

    """
    # Plot the variation
    ax.hist(variations, color="black", histtype="step")
    # Draw a line at the mean value
    ax.axvline(mean, color="brown")
    # Add some annotations
    ax.annotate(f"{len(variations)} variations", (0.69, 0.9), xycoords="axes fraction")
    ax.annotate(string, (0.69, 0.85), xycoords="axes fraction")
