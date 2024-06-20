from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sysvar.corrections import BaseCorrection, CorrectionBF
    from sysvar.uncertainties import Uncertainty
    from sysvar.variations import Variator
    from sysvar.templates import Template
    from sysvar.eigendecomposer import EigenDecomposer

from datetime import datetime
from os import path, makedirs
import math

from typing import Union, Iterable

from abc import ABC, abstractmethod

import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm


PALETTE = sns.color_palette("colorblind")
CMAP = "Blues"


class Visualizer(ABC):
    def __init__(
        self,
        instance: BaseCorrection,
        namespace: list = None,
        top_dir: str = None,
        dir_spec: Union[str, None, bool] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):

        self.instance = instance
        self.namespace = namespace
        self.top_dir = top_dir
        # Get the save dir. By default this is today's date
        self.save_dir = self._get_save_dir(top_dir, dir_spec)
        self.extensions = self._get_extensions(extra_ext)
        self.save = save
        super().__init__()

    def create_single_figure(self):
        return plt.subplots(figsize=(8, 5), dpi=800)

    def available_plots(
        self,
    ):
        raise NotImplementedMethod(
            "Implement a method that will tell you what kind of plots are available"
        )

    @abstractmethod
    def plot_cov_matrix(self):
        pass

    @abstractmethod
    def plot_corr_matrix(self):
        pass

    def plot_cov_and_corr(self):

        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), dpi=800, sharey=True)

        self.plot_cov_matrix(ax[0])
        self.plot_corr_matrix(ax[1])

        self.annotate_matrix_plot(fig, ax)

        if self.save:
            self.save_figure(fig, ["cov_and_corr"])

        return fig, ax

    @abstractmethod
    def annotate_matrix_plot(self):
        pass

    def save_figure(
        self,
        fig: Figure,
        fig_name_comps: list,
    ):
        """Helper function to save figure when it is called.
        The figure needs to be passed as a argument.
        The name of the figure is combined by default with png and pdf extensions and then saved

        Args:
        fig: plt figure to save
        fig_name_comps: components to combine into a single figure name

        """
        # First check if the dir exists already
        self.check_if_dir_exists()

        # build the name of the figure
        name = "_".join((self.namespace + fig_name_comps))

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
            makedirs(self.save_dir)

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
    def _get_save_dir(top_dir, dir_spec):

        if dir_spec:
            today = datetime.today().strftime("%Y-%m-%d")
            dir_name = today if dir_spec is None else "-".join((today, dir_spec))

            outdir = path.join(top_dir, dir_name)
        else:
            outdir = top_dir

        return outdir

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


class NoMatrixError(Exception):
    pass


class CorrectionVisualizer(Visualizer):
    def __init__(
        self,
        instance: BaseCorrection,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)

    def annotate_matrix_plot(self, ax: Axes):
        raise NoMatrixError(
            "The Correction object does not have a covariance nor a correlation matrix. This is normal! Don't try to call this method on this class"
        )

    def plot_cov_matrix(self):
        raise NoMatrixError(
            "The Correction object does not have a covariance nor a correlation matrix. This is normal! Don't try to call this method on this class"
        )

    def plot_corr_matrix(self):
        raise NoMatrixError(
            "The Correction object does not have a covariance nor a correlation matrix. This is normal! Don't try to call this method on this class"
        )

    def plot_error_comparison(self):

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
            np.arange(len(self.instance.central_values)), self.instance.visual_labels
        )
        ax.set_xlabel("Correction weight")
        ax.set_title(f"{self.instance.title} uncertainties")
        plt.legend(bbox_to_anchor=(1, 0.7))

        if self.save:
            self.save_figure(fig, ["error_comparison"])

        return fig, ax


class UncertaintyVisualizer(Visualizer):
    def __init__(
        self,
        instance: Uncertainty,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)

    def annotate_matrix_plot(self, fig: Figure, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(axis)

        fig.suptitle(self.instance.name + " uncertainty")

    def _annotate_axis(self, ax):

        ax.set_xlabel("Correction bins")
        ax.set_ylabel("Correction bins")

        ax.set_xticks(
            np.arange(len(self.instance.string_boundaries)) + 0.5,
            self.instance.string_boundaries,
            rotation=30,
        )
        ax.set_yticks(
            np.arange(len(self.instance.string_boundaries)) + 0.5,
            self.instance.string_boundaries,
            rotation=0,
        )

    def plot_cov_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Covariance"},
            cmap=CMAP,
        )
        ax.set_title("Covariance matrix")

        return ax

    def plot_corr_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.corr_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Correlation matrix")

        return ax


class VariatorVisualizer(Visualizer):
    def __init__(
        self,
        instance: Variator,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)
        self._strings = self.instance.correction.visual_labels

    @property
    def strings(self) -> list:
        return self._strings

    @strings.setter
    def strings(self, values):
        self._strings = values

    def annotate_matrix_plot(self, fig: Figure, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(axis)

        fig.suptitle(f"Total covariance for {self.instance.correction.title}")

    def _annotate_axis(self, ax):

        ax.set_xlabel("Correction bins")
        ax.set_ylabel("Correction bins")

        # PATCH
        # FIXME
        # Add local import to avoid circular imports
        from sysvar.corrections import CorrectionBF

        ax.set_xticks(
            np.arange(len(self._strings)) + 0.5,
            self._strings,
            rotation=(90 if isinstance(self.instance.correction, CorrectionBF) else 0),
        )
        ax.set_yticks(
            np.arange(len(self._strings)) + 0.5,
            self._strings,
            rotation=(0 if isinstance(self.instance.correction, CorrectionBF) else 0),
        )

    def plot_cov_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Covariance"},
            cmap=CMAP,
        )
        ax.set_title("Covariance matrix")

        return ax

    def plot_corr_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.corr_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Correlation matrix")

        return ax

    def plot_gaussian_variations(self):
        """
        Plot Gaussian variations of the corrections.

        Args:

        Returns:
            None

        """

        fig, ax = plt.subplots(
            1, len(self.instance.correction.central_values), figsize=(16, 4.5), dpi=800
        )

        for i, (mean, s) in enumerate(
            zip(self.instance.correction.central_values, self._strings)
        ):

            # Plot the variation
            ax[i].hist(self.instance.variations[:, i], color="black", histtype="step")

            # Draw a line at the mean value
            ax[i].axvline(mean, color="brown")

            ax[i].set_title(s, fontsize=14)

        fig.suptitle(f"{self.instance.Nvar} variations", fontsize=18)

        if self.save:
            self.save_figure(fig, ["gaussian_variations"])

        return fig, ax

    def plot_relative_variations(self, value_edges: Iterable, Nvar: int = 5):
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
                x=value_edges,
                variation=self.instance.variations[i, :]
                / self.instance.correction.central_values,
                index=i,
                plot_func="stairs",
            )

        if self.save:
            self.save_figure(fig, ["relative_variations"])

        return fig, ax

    def plot_relative_variations_in_grid(self, nbins: int = 41):

        counts = []
        bin_edges = []

        min_var = np.round(np.min(self.instance.relative_variations), 2)
        max_var = np.round(np.max(self.instance.relative_variations), 2)

        for i in range(len(self.instance.correction.central_values)):
            hist = np.histogram(
                self.instance.relative_variations[:, i],
                range=(min_var, max_var),
                bins=nbins,
            )
            counts.append(hist[0])
            bin_edges.append(hist[1])

        fig, ax = plt.subplots(figsize=(5, 10))

        cb = ax.matshow(np.array(counts).T, cmap=CMAP)
        plt.colorbar(cb)

        ax.set_xticks(np.arange(len(self.strings)), self.strings, rotation=90)
        ax.set_yticks(
            np.arange(len(bin_edges[0][:-1])),
            np.round((bin_edges[0][1:] + bin_edges[0][:-1]) / 2, 3),
        )
        ax.set_ylabel("Relative variation")

        ax.invert_yaxis()

        if self.save:
            self.save_figure(fig, ["relative_variations_in_grid"])

        return fig, ax


class TemplateVisualizer(Visualizer):
    def __init__(
        self,
        instance: Template,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)

    def annotate_matrix_plot(self, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(ax)

    def _annotate_axis(self, ax):

        ax.set_xlabel("Bins")
        ax.set_ylabel("Bins")

    def plot_relative_variations_in_grid(self, ax, nbins: int = 11):

        counts = []
        bin_edges = []

        variations = self.instance._get_absolute_variations()
        nominals = self.instance.make_hist()

        relative_variations = np.nan_to_num(variations.T / nominals[0], nan=1)

        min_var = np.round(np.min(relative_variations), 2)
        max_var = np.round(np.max(relative_variations), 2)

        for i in range(relative_variations.shape[1]):
            hist = np.histogram(
                relative_variations[:, i],
                range=(min_var, max_var),
                bins=nbins,
            )
            counts.append(hist[0])
            bin_edges.append(hist[1])

        cb = ax.matshow(np.array(counts).T, cmap=CMAP)
        plt.colorbar(cb)

        ax.set_xlabel("fit variable bins")
        ax.set_xticks(
            np.arange(relative_variations.shape[1]),
            np.arange(relative_variations.shape[1]) + 1,
        )
        ax.set_yticks(
            np.arange(nbins),
            np.round(np.linspace(min_var, max_var, nbins), 2),
        )
        ax.set_ylabel("Relative variation")

        ax.invert_yaxis()

        if self.save:
            self.save_figure(fig, ["template_relative_variations_in_grid"])

        return ax

    def plot_cov_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Covariance"},
            cmap=CMAP,
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
            self.instance.corr_matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Correlation matrix")

        return ax

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

        if self.save:
            try:
                self.save_figure(fig, ["nominal_template"])
            except UnboundLocalError:
                # Don't save the plot if this is a part of a bigger plot
                pass

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

        if self.save:
            self.save_figure(fig, [f"{str(Nvar)} variations"])

        return fig, ax

    def plot_up_and_down_variations(self, ax: Union[np.ndarray, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        x = np.linspace(0, 1, self.instance.Nbins)

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
        ax[0].legend()

        ax[1].plot(
            x,
            (h_up[0].flatten() - self.instance.nom_hist[0].flatten())
            / (np.sqrt(self.instance.nom_hist[0].flatten())),
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

        if self.save:
            try:
                self.save_figure(fig, [f"up_and_down_variations"])
            except UnboundLocalError:
                # Don't save the plot if this is a part of a bigger plot
                pass

    def plot_eigenvalues(self, ax: Union[np.ndarray, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        x = np.arange(self.instance.Nbins)

        ax.plot(x, self.instance.eigen_values, linestyle="", marker=".", color="black")

        ax.set_yscale("log")
        secax = ax.secondary_xaxis(
            "top",
        )
        ax.set_xticks(x, [])
        ax.set_ylabel("Eigenvalues")
        ax.set_xlabel("Eigendirection")

        eigenvalue_ticks = [
            (
                round(x, 3)
                if x > 10e-3
                else rf"~$10^{{{math.floor(math.log(abs(x), 10))}}}$"
            )
            for x in self.instance.eigen_values
        ]
        secax.set_xticks(x, eigenvalue_ticks, fontsize=10)

        for ticklabel, egv_tick in zip(secax.get_xticklabels(), eigenvalue_ticks):
            if isinstance(egv_tick, float):
                ticklabel.set_color("#07529a")
                ticklabel.set_fontsize(10)
                ticklabel.set_rotation(90)
            else:
                ticklabel.set_color("grey")
                ticklabel.set_fontsize(7)

    def plot_systematic_overview(self):

        # gridspec inside gridspec
        fig = plt.figure(figsize=(16, 10), dpi=800)

        gs = mpl.gridspec.GridSpec(2, 6, wspace=0.4, hspace=0.15)

        ax0 = fig.add_subplot(gs[0, 0:4])
        self.plot_nominal_template(ax0)

        self.annotate_matrix_plot(ax0)

        ax01 = fig.add_subplot(gs[0, 4:6])
        # self.plot_eigenvalues(ax01)
        self.plot_relative_variations_in_grid(ax01)

        ax1 = fig.add_subplot(gs[1, :3])
        self.plot_corr_matrix(ax1)

        gs_low = gs[1, 3:].subgridspec(2, 1, height_ratios=[3.5, 1], hspace=0.1)
        ax2 = fig.add_subplot(gs_low[0, 0])
        ax2.set_xticks([])
        ax3 = fig.add_subplot(gs_low[1, 0])
        self.plot_up_and_down_variations(np.array([ax2, ax3]))

        if self.save:
            self.save_figure(fig, ["systematic_overview"])

        return fig, (ax0, ax01, ax1, ax2, ax3)


class FFModelVisualizer(Visualizer):
    def __init__(
        self,
        instance: Template,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)

    def annotate_matrix_plot(self, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(axis)

    def _annotate_axis(self, ax):

        ax.set_xlabel("FF model parameters")
        ax.set_ylabel("FF model parameters")

        ax.set_xticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )
        ax.set_yticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )

    def plot_cov_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Covariance"},
            cmap=CMAP,
        )
        ax.set_title("Covariance matrix")
        self.annotate_matrix_plot(ax)

        return ax

    def plot_corr_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.corr_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            " ".join(
                (
                    "Correlation matrix",
                    self.instance.name,
                    r"$\mathrm{B} \rightarrow$",
                    get_latex_symbol(self.instance.Xc),
                    get_latex_symbol(self.instance.lep),
                    get_latex_symbol("Nu"),
                )
            )
        )

        return ax

    def plot_params(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        ax.errorbar(
            x=np.arange(self.instance.num_params) + 0.5,
            y=self.instance.parameter_central_values,
            yerr=self.instance.parameter_errors,
            linestyle="",
            color="black",
            marker=".",
        )

        ax.table(
            cellText=[
                [round(x, 4) for x in self.instance.parameter_central_values],
                [round(x, 4) for x in self.instance.parameter_errors],
            ],
            rowLabels=["value", "unc"],
            colLabels=[
                x if x != "DelMbc" else "DelMbc(mc)"
                for x in self.instance.params.keys()
            ],
            loc="top",
        )

        ax.set_xticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )
        ax.set_xlabel("FF model parameters")
        ax.set_ylabel("Value")
        # ax.set_title("Model parameter values")

        return ax

    def plot_corr_and_params(self):

        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), dpi=800)

        self.plot_params(ax[0])

        self.plot_corr_matrix(ax[1])
        self.annotate_matrix_plot(ax[1])

        return fig, ax


class EigenDecomposerVisualizer(Visualizer):
    def __init__(
        self,
        instance: EigenDecomposer,
        namespace: list,
        top_dir: str,
        dir_spec: Union[str, None] = None,
        extra_ext: Union[str, Iterable, None] = None,
        save: bool = False,
    ):
        super().__init__(instance, namespace, top_dir, dir_spec, extra_ext, save)

    def plot_cov_matrix(self):
        pass

    def annotate_matrix_plot(self, ax: Axes):

        if isinstance(ax, Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(ax)

    def _annotate_axis(self, ax):

        ax.set_xlabel("Templates/Bins")
        ax.set_ylabel("Templates/Bins")

    def plot_corr_matrix(self, ax: Union[Axes, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        sns.heatmap(
            self.instance.corr,
            ax=ax,
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Correlation matrix")

        return ax

    def plot_eigenvalues(self, ax: Union[np.ndarray, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        x = np.arange(self.instance.eigen_values.shape[0])

        ax.plot(x, self.instance.eigen_values, linestyle="", marker=".", color="black")

        ax.set_yscale("log")
        ax.set_ylabel("Eigenvalues")
        ax.set_xlabel("Eigendirection")

        ax.text(
            x[-1] / 1.5,
            self.instance.precision + 0.02,
            f"Arb. threshold set: {self.instance.precision}",
            color="#07529aff",
        )

        ax.text(
            x[-1] / 2,
            np.max(self.instance.eigen_values),
            f"Keeping {self.instance.N_important_dims}/{x[-1]+1} eigendirections",
            color="#eab90cff",
        )

        ax_right = ax.twinx()

        # FIXME this is wrong and misleading.
        # The two axes are not aligned
        ax_right.plot(
            x,
            self.instance.max_differences / self.instance.cov.max(),
            color="white",
            alpha=0,
        )
        ax_right.axhline(self.instance.precision, color="#07529aff")
        ax_right.set_yscale("log")
        ax_right.set_ylabel(r"max($\frac{|Cov - Cov^{'}|}{Cov}$)", color="#07529aff")

    def plot_cov_diff(self, ax: Union[np.ndarray, None] = None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=800)

        total_N_eigendirections = self.instance.eigen_values.shape[0]
        x = np.arange(len(self.instance.max_differences))
        y = self.instance.max_differences / self.instance.cov.max()
        ax.plot(
            x,
            y,
            linestyle="",
            marker=".",
            color="black",
        )

        ax.set_yscale("log")
        ax.set_ylabel(r"max($\frac{|Cov - Cov^{'}|}{Cov}$)")
        ax.set_xlabel("Eigendirection")

        ax.annotate(
            f"Keeping {self.instance.N_important_dims}/{total_N_eigendirections} eigendirections \n (first 50 considered only)",
            (0.5, 0.5),
            xycoords="axes fraction",
            color="#eab90cff",
        )

        ax.fill_between(
            [-10, x[-1]], self.instance.precision, 1, alpha=0.5, color="#07529aff"
        )


def get_latex_symbol(key):

    dictionary = {
        "D**0": r"$\mathrm{D^{**}_{0}}$",
        "D**0*": r"$\mathrm{D^{**}_{0'}}$",
        "D**1": r"$\mathrm{D^{**}_{1}}$",
        "D**1*": r"$\mathrm{D^{**}_{1'}}$",
        "D**2": r"$\mathrm{D^{**}_{2}}$",
        "D**2*": r"$\mathrm{D^{**}_{2'}}$",
        "D*": r"$\mathrm{D^{*}}$",
        "D": "D",
        "Tau": r"$\mathrm{\tau}$",
        "Ell": r"$\mathrm{\ell}$",
        "Nu": r"$\mathrm{\nu}$",
    }

    return dictionary[key]
