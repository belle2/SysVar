from datetime import datetime
from os import path, mkdir

from typing import Union, Iterable

from abc import ABC, abstractmethod

import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

from sysvar.uncertainties import Uncertainty
from sysvar.variations import Variator
from sysvar.corrections import Correction

PALETTE = sns.color_palette("colorblind")


def create_single_figure():
    return plt.subplots(figsize=(8, 5), dpi=800)


def create_double_figure():
    return plt.subplots(1, 2, figsize=(16, 4.5), dpi=800)


def create_triple_figure():
    return plt.subplots(1, 3, figsize=(16, 4.5), dpi=800)


class Visualizer(ABC):
    def __init__(
        self,
        instance: Union[Correction, Uncertainty, Variator, Template],
        namespace: list,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
    ):

        self.instance = instance
        self.namespace = namespace
        # Get the save dir. By default this is today's date
        self.save_dir = self._get_save_dir(dir_spec)
        self.extensions = self._get_extensions(extra_ext)

    @abstractmethod
    def annotate_matrix_plot(self, ax: Axes):
        pass

    def plot_cov_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            fmt=".2f",
            cbar_kws={"label": "Covariance"},
            cmap="Blues",
            norm=LogNorm(),
            vmin=0.0001,
            vmax=100,
        )
        ax.set_title("Covariance matrix")

        return ax

    def plot_corr_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            cbar_kws={"label": "Pearson coeff."},
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        ax.set_title("Correlation matrix")

        return ax

    def plot_cov_and_corr(self, ax: np.ndarray):

        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), dpi=800)

        self.plot_cov_matrix(ax[0])
        self.plot_corr_matrix(ax[1])

        return fig, ax

    def save_figure(
        self,
        fig: Figure,
        fig_name_comps: str,
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
        # First check if the dir exists already
        self.check_if_dir_exists()

        # build the name of the figure
        name = "_".join(self.namespace.extend(fig_name_comps))

        # Loop over the extensions, create the figname and then save the figure
        for ext in self.extensions:

            # Check if ext alread has dot in the beginning. If not add it
            if ext[0] != ".":
                ext.insert(0, ".")

            figname = name + ext
            fig.savefig(path.join(self.save_dir, figname), bbox_inches="tight", dpi=800)
            print(f"Saved figures in {self.save_dir}")

    def check_if_dir_exists(self):

        if path.exists(self.save_dir):
            pass
        else:
            mkdir(self.save_dir)

    @staticmethod
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

    @staticmethod
    def _get_save_dir(dir_spec):

        top_dir = "/nfs/dust/belle2/user/itsaklid/analysis/figures"
        today = datetime.today().strftime("%Y-%m-%d")

        dir_name = today if dir_spec is None else "_".join((today, dir_spec))

        return path.join(top_dir, dir_name)


class CorrectionVisualizer(Visualizer):
    def __init__(
        self,
        instance: Correction,
        namespace: list,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
    ):
        super().__init__(instance, namespace, dir_spec, extra_ext)

    def plot_error_comparison_in_axis(self):

        fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        # Plot the central values of the correction
        ax.errorbar(
            self.instance.central_values,
            np.arange(len(self.instance.central_values)),
            xerr=self.instance.total_error,
            linestyle="",
            marker="o",
            color="black",
            label="central value w/ total unc.",
            capsize=5,
        )

        for i, (n, u) in enumerate(reversed(self.instance.uncertainties.items())):
            ax.errorbar(
                self.instance.central_values,
                np.arange(len(u.errors)) + (i + 1) * 0.1,
                xerr=u.errors,
                label=n,
                linestyle="",
                capsize=5,
                color=PALETTE[i],
            )

        ax.set_yticks(
            np.arange(len(self.instance.central_values)), self.instance.strings
        )
        ax.set_xlabel("Correction weight")
        plt.legend(bbox_to_anchor=(1, 0.7))

        return

    @staticmethod
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


class UncertaintyVisualizer(Visualizer):
    def __init__(
        self,
        instance: Uncertainty,
        namespace: list,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
    ):
        super().__init__(instance, namespace, dir_spec, extra_ext)

    def annotate_matrix_plot(self, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(ax)

    def _annotate_axis(self, ax):

        ax.set_xlabel("Correction bins")
        ax.set_ylabel("Correction bins")

        ax.set_xticks(
            np.arange(len(self.instance.string_boundaries)),
            self.instance.string_boundaries,
        )
        ax.set_yticks(
            np.arange(len(self.instance.string_boundaries)),
            self.instance.string_boundaries,
        )


class VariationVisualizer(Visualizer):
    def __init__(
        self,
        instance: Variator,
        namespace: list,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
    ):
        super().__init__(instance, namespace, dir_spec, extra_ext)

    def plot_gaussian_variations(self, strings: list):

        """
        Plot Gaussian variations of the corrections.

        Args:
            strings (list): A list of the strings of the range of the dependant variables.

        Returns:
            None

        """

        fig, ax = plt.subplots(
            1, len(self.correction.central_values), figsize=(16, 4.5), dpi=800
        )

        for i, (mean, s) in enumerate(zip(self.instance.central_values, strings)):

            # Plot the variation
            ax[i].hist(self.instance.variations, color="black", histtype="step")

            # Draw a line at the mean value
            ax.axvline(mean, color="brown")
            # Add some annotations
            ax.annotate(
                f"{len(self.instance.Nvar)} variations",
                (0.69, 0.9),
                xycoords="axes fraction",
            )
            ax.annotate(s, (0.69, 0.85), xycoords="axes fraction")

        return fig, ax

    def plot_relative_variations(self, Nvar: int = 5):

        """
        Plots the relative variations of the templates.
        The Nvar argument specifies the number of variatios that will be plotted.
        Defaults to 5.

        Args:
            Nvar (int, optional): The number of variations to visualize.

        Returns:
            Tuple[Figure, Axis]: A tuple containing the figure and axis objects.

        """

        fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        for i in range(Nvar):
            self.plot_variation_on_axis(
                ax=ax,
                x=self.correction.value_edges,
                variation=self.instance.variations[i, :] / self.instance.central_values,
                index=i,
                plot_func="stairs",
            )

        return fig, ax


class TemplateVisualizer(Visualizer):
    def __init__(
        self,
        instance: Variator,
        namespace: list,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
    ):
        super().__init__(instance, namespace, dir_spec, extra_ext)

    def plot_nominal_template(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        self.plot_variation_on_axis(
            ax,
            np.linspace(0, 1, self.instance.Nbins + 1),
            self.instance.nom_hist[0].flatten(),
            plot_func="stairs",
        )

        ax.set_ylabel("Events / bin")
        ax.set_xlabel("Fitting variable")

        return ax

    def plot_variations(self, Nvar: int = 5):

        fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        for i in range(Nvar):

            v_hist = self.instance.make_hist(index=i)

            bin_edges = [np.array(b) for b in self.instance.binning.values()]
            x = np.linspace(0, 1, self.instance.Nbins)

            self.plot_variation_on_axis(
                ax, x, v_hist[0].flatten() / self.instance.nom_hist[0].flatten(), i
            )

        ax.set_ylabel("Template relative variation")
        ax.set_xlabel("Fitting variable")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax

    def plot_up_and_down_variations(self, ax: Union[np.ndarray, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        x = np.linspace(0, 1, self.Nbins)

        h_up = self.instance.make_hist("up")
        h_down = self.instance.make_hist("down")

        ax[0].axhline(y=1, color="black")

        ax[0].step(
            x,
            h_up[0].flatten() / self.instance.nom_hist[0].flatten(),
            color=PALETTE[0],
            label="Up variation",
            linestyle="dashed",
        )

        ax[0].step(
            x,
            h_down[0].flatten() / self.instance.nom_hist[0].flatten(),
            color=PALETTE[2],
            label="Down variation",
            linestyle="dashed",
        )

        ax[0].fill_between(
            x,
            1
            - np.sqrt(self.instance.nom_hist[0].flatten())
            / self.instance.nom_hist[0].flatten(),
            1
            + np.sqrt(self.instance.nom_hist[0].flatten())
            / self.instance.nom_hist[0].flatten(),
            color="grey",
            alpha=0.25,
            label="Stat error",
        )

        ax[0].set_ylabel("Template relative variation")
        ax[0].set_xlabel("Fitting variable")
        ax[0].legend()

        ax[1].plot(
            x,
            (h_up[0].flatten() - self.nom_hist[0].flatten())
            / (np.sqrt(self.nom_hist[0].flatten())),
            linestyle="",
            marker=".",
            color="black",
        )

        ax[1].fill_between(x=x, y1=10e-3, y2=10e-2, color=PALETTE[0], alpha=0.75)
        ax[1].fill_between(x=x, y1=10e-2, y2=10e-1, color=PALETTE[3], alpha=0.75)

        ax[1].set_ylabel("variation/stat error")
        ax[1].set_xlabel("Fitting variable")
        ax[1].set_ylim(0.01, 1)
        ax[1].set_yscale("log")

    def plot_systematic_overview(self):

        # gridspec inside gridspec
        fig = plt.figure(figsize=(16, 10), dpi=800)

        gs = mpl.gridspec.GridSpec(2, 6, wspace=0.4, hspace=0.15)
        ax0 = fig.add_subplot(gs[0, 1:5])
        ax1 = fig.add_subplot(gs[1, :3])
        gs_low = gs[1, 3:].subgridspec(2, 1, height_ratios=[3.5, 1], hspace=0.1)

        ax2 = fig.add_subplot(gs_low[0, 0])
        ax2.set_xticks([])
        ax3 = fig.add_subplot(gs_low[1, 0])

        self.plot_corr_matrix(ax0)

        self.plot_nominal_template(ax1)

        self.plot_up_and_down_variations(np.array([ax2, ax3]))

        return fig, (ax0, ax1, ax2, ax3)
