from datetime import datetime
from os import path, mkdir

from typing import Union

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PALETTE = sns.color_palette("colorblind")


def create_single_figure():
    return plt.subplots(figsize=(8, 5), dpi=800)


def create_double_figure():
    return plt.subplots(1, 2, figsize=(16, 4.5), dpi=800)


def create_triple_figure():
    return plt.subplots(1, 3, figsize=(16, 4.5), dpi=800)


def plot_matrix_on_axis(ax, matrix, tick_labels, title, axes_labels):

    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax, vmin=0, vmax=1)

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


def plot_error_comparison_in_axis(ax: Axes, correction):

    ax.errorbar(
        correction.central_values,
        np.arange(len(correction.central_values)),
        xerr=correction.total_error,
        linestyle="",
        marker="o",
        color="black",
        label="central value w/ total unc.",
        capsize=5,
    )

    for i, (n, u) in enumerate(reversed(correction.uncertainties.items())):
        ax.errorbar(
            correction.central_values,
            np.arange(len(u.errors)) + (i + 1) * 0.1,
            xerr=u.errors,
            label=n,
            linestyle="",
            capsize=5,
            color=PALETTE[i],
        )

    ax.set_yticks(np.arange(len(correction.central_values)), correction.strings)
    ax.set_xlabel("correction weight")
    plt.legend(bbox_to_anchor=(1, 0.7))

    return


def save_figure(
    fig: Figure,
    fig_name_comps: str,
    dir_spec: Union[str, None] = None,
    extra_ext: Union[str, list, None] = None,
):
    """Helper function to save figure when it is called.
    The figure needs to be passed as a argument.
    The name of the figure is combined by default with png and pdf extensions and then saved
    Extra extensions can be passed as arguments

    Args:
    fig: plt figure to save
    fig_name_comps: components to combine into a single figure name
    dir_spec: name specifier for the directory to be added after the date.
              Defaults to None
    extra_ext: extra extension, on top of pdf and png to save the figure with.
    """
    # Get the save dir. By default this is today's date
    save_dir = _get_save_dir(dir_spec)
    # First check if the dir exists already
    check_if_dir_exists(save_dir)
    extensions = _get_extensions(extra_ext)

    # build the name of the figure
    name = "_".join(fig_name_comps)

    # Loop over the extensions, create the figname and then save the figure
    for ext in extensions:

        # Check if ext alread has dot in the beginning. If not add it
        if ext[0] != ".":
            ext.insert(0, ".")

        figname = name + ext
        fig.savefig(path.join(save_dir, figname), bbox_inches="tight", dpi=800)
    print(f"Saved figures in {save_dir}")


def _get_extensions(extra_ext):

    # Define the defaule extensions
    extensions = [".pdf", ".png"]
    if extra_ext is not None:

        if isinstance(extra_ext, str):
            extensions.append(extra_ext)
        elif isinstance(extra_ext, list):
            extensions.extend(extra_ext)
        else:
            raise TypeError(f"Unkown type {type(extra_ext)} for extra_ext")

    return extensions


def _get_save_dir(dir_spec: Union[str, None] = None):

    top_dir = "/nfs/dust/belle2/user/itsaklid/analysis/figures"
    today = datetime.today().strftime("%Y-%m-%d")

    dir_name = today if dir_spec is None else "_".join((today, dir_spec))

    return path.join(top_dir, dir_name)


def check_if_dir_exists(path_to_dir):

    if path.exists(path_to_dir):
        pass
    else:
        mkdir(path_to_dir)
