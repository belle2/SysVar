from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from pandas import DataFrame
import seaborn as sns
from matplotlib.colors import LogNorm

from sysvar.visualize import (
    plot_matrix_on_axis,
    plot_variation_on_axis,
    create_double_figure,
    create_single_figure,
)
from sysvar.corrections import Correction
from sysvar.variations import Variator


class NotADictError(Exception):
    pass


class NotCorrectVariableError(Exception):
    pass


class NotIncreasingBinning(Exception):
    pass


class Template(ABC):
    def __init__(
        self,
        df: DataFrame,
        binning: dict,
        total_weight: str,
        syst_weight: str,
        correction: Correction,
        variator: Variator,
    ):
        if self._is_correct_binning(df.columns, binning):
            self.binning = binning
        if self._is_existing_variable(total_weight, df.columns):
            self.total_weight = total_weight
        if self._is_existing_variable(syst_weight, df.columns):
            self.syst_weight = syst_weight
        # Make a deep copy only of the columns that are needed
        self.df = df[
            [
                *binning.keys(),
                self.total_weight,
                self.syst_weight,
                correction.dependant_variable,
            ]
        ].copy(deep=True)

        self.correction = correction
        self.variations = variator.variations
        self.Nvar = variator.Nvar

    def _is_correct_binning(self, columns: list, binning: dict) -> bool:

        if self._is_a_dict(binning):
            pass
        # Check if the variables exist in the dataframe
        for var_name in binning.keys():
            if self._is_existing_variable(var_name, columns):
                continue
        for var_name, bins in binning.items():
            if self._is_increasing_binning(var_name, bins):
                continue

        return True

    @staticmethod
    def _is_a_dict(binning: dict) -> bool:
        # Check of the binning is a dictionary
        if not isinstance(binning, dict):
            raise NotADictError(
                "The binning argument should be a dictionary with the names of the variables and their corresponding binning e.g.  {var1: [0, 1, 2], var2: [0.1, 0.2, 0.3]}"
            )
        else:
            return True

    @staticmethod
    def _is_existing_variable(var_name: str, columns: list) -> bool:

        if var_name not in columns:
            raise NotCorrectVariableError(
                f"{var_name} does not exist in the dataframe columns"
            )
        return True

    @staticmethod
    def _is_increasing_binning(var_name, bins) -> bool:

        # Ensure that bins is a np array
        bins_array = np.array(bins)
        diff = np.diff(bins_array)
        if not np.all(diff > 0):
            raise NotIncreasingBinning(
                f"The binning for variable {var_name} is not strictly increasing"
            )

        return True

    def _get_number_of_bins(self):
        return np.prod([len(bins) - 1 for bins in self.binning.values()])

    @abstractmethod
    def make_hist(self, index: Union[None, int] = None) -> np.ndarray:
        pass

    def add_variations(self):

        # Initialize the variations
        self.df.loc[:, [f"{self.syst_weight}_var_{i}" for i in range(self.Nvar)]] = 1

        for i, q in enumerate(self.correction.queries):
            # Now add the variations of the corrections to the dataframe entries that
            # pass the cuts
            self.df.loc[
                self.df.eval(q),
                [f"{self.syst_weight}_var_{j}" for j in range(self.Nvar)],
            ] = self.variations[:, i]

    def _get_absolute_variations(self):
        absolute_variations = np.empty((self._get_number_of_bins(), self.Nvar))
        for i in range(self.Nvar):
            absolute_variations[:, i] = self.make_hist(i)[0].flatten()

        return absolute_variations

    def get_bin_covariance(self) -> np.ndarray:

        return np.cov(self._get_absolute_variations())

    def get_bin_correlation(self) -> np.ndarray:

        return np.corrcoef(self._get_absolute_variations())

    def visualize_bin_covariance(self):

        fig, ax = create_single_figure()

        plot_matrix_on_axis(
            ax,
            self.get_bin_covariance(),
            np.arange(self._get_number_of_bins()),
            "Covariance matrix",
            "bins",
        )

        return fig, ax

    def visualize_bin_correlation(self):

        fig, ax = create_single_figure()

        plot_matrix_on_axis(
            ax,
            self.get_bin_correlation(),
            np.arange(self._get_number_of_bins()),
            "Correlation matrix",
            "bins",
        )

        return fig, ax

    def visualize_bin_covariance_and_correlation(self):

        fig, ax = create_double_figure()

        sns.heatmap(
            self.get_bin_covariance(),
            annot=True,
            ax=ax[0],
            fmt=".2f",
            cbar_kws={"label": "Covariance"},
            cmap="Blues",
            norm=LogNorm(),
            vmin=0.0001,
            vmax=100,
        )

        sns.heatmap(
            self.get_bin_correlation(),
            annot=True,
            ax=ax[1],
            cbar_kws={"label": "Pearson coeff."},
            cmap="Blues",
            vmin=0,
            vmax=1,
        )

        return fig, ax


class Template1D(Template):
    def __init__(
        self, df: DataFrame, binning: dict, total_weight: str, syst_weight: str
    ):
        raise NotImplementedError("Only 2D histograms supported  now")


class Template2D(Template):
    def __init__(
        self,
        df: DataFrame,
        binning: dict,
        total_weight: str,
        syst_weight: str,
        correction: Correction,
        variator: Variator,
    ):
        super().__init__(df, binning, total_weight, syst_weight, correction, variator)

        self.nom_hist = self.make_hist()

    def make_hist(self, index: Union[None, int] = None) -> np.ndarray:

        if index is None:
            # Take the nominal total weight
            weights = np.array(self.df[self.total_weight])
        else:
            # Divide with the nominal systematic weight and multiply with the varied one
            weights = np.array(
                (self.df[self.total_weight] / self.df[self.syst_weight])
                * self.df[f"{self.syst_weight}_var_{index}"]
            )

        return np.histogramdd(
            np.array(self.df[[*self.binning.keys()]]),
            bins=[bins for bins in self.binning.values()],
            weights=weights,
        )

    def visualize_nominal_template(self):

        fig, ax = create_single_figure()

        plot_variation_on_axis(
            ax,
            np.linspace(0, 1, self._get_number_of_bins() + 1),
            self.nom_hist[0].flatten(),
            plot_func="stairs",
        )

        ax.set_ylabel("Events / bin")
        ax.set_xlabel("Fitting variable")

        return fig, ax

    def visualize_variations(self, Nvar: int = 5):

        fig, ax = create_single_figure()

        for i in range(Nvar):

            v_hist = self.make_hist(index=i)

            bin_edges = [np.array(b) for b in self.binning.values()]
            x = np.linspace(0, 1, self._get_number_of_bins())

            plot_variation_on_axis(
                ax, x, v_hist[0].flatten() / self.nom_hist[0].flatten(), i
            )

        ax.set_ylabel("Template relative variation")
        ax.set_xlabel("Fitting variable")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax
