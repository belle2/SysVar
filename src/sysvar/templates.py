from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from pandas import DataFrame

from sysvar.visualize import (
    plot_matrix_on_axis,
    plot_variation_on_axis,
    create_double_figure,
    create_single_figure,
)


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
        dependant_variables: list[str],
    ):
        if self._is_correct_binning(df.columns, binning):
            self.binning = binning
        if self._is_existing_variable(total_weight, df.columns):
            self.total_weight = total_weight
        if self._is_existing_variable(syst_weight, df.columns):
            self.syst_weight = syst_weight
        # Make a deep copy only of the columns that are needed
        self.df = df[
            [*binning.keys(), self.total_weight, self.syst_weight, *dependant_variables]
        ].copy(deep=True)

    def _is_correct_binning(self, columns: list, binning: dict) -> bool:

        if self._is_a_dict(columns, binning):
            pass
        # Check if the variables exist in the dataframe
        for var_name in binning.keys():
            if self._is_existing_variable(var_name, columns):
                continue
        for var_name, bins in binning.keys():
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

    @abstractmethod
    def make_hist(self) -> np.ndarray:
        pass

    def add_variations(self, queries: list, variations: np.ndarray, Nvar: int):

        # Initialize the variations
        self.df[[f"{self.syst_weight}_var_{i}" for i in range(Nvar)]] = 1

        for i, q in enumerate(queries):
            # Now add the variations of the corrections to the dataframe entries that
            # pass the cuts
            self.df.loc[
                self.df.eval(q), [f"{self.syst_weight}_var_{i}" for i in range(Nvar)]
            ] = variations[:, i]


class Template1D(Template):
    def __init__(
        self, df: DataFrame, binning: dict, total_weight: str, syst_weight: str
    ):
        raise NotImplementedError("Only 2D histograms supported  now")


class Template2D(Template):
    def __init__(
        self, df: DataFrame, binning: dict, total_weight: str, syst_weight: str
    ):
        super.__init__(df, binning, total_weight, syst_weight)

        self.nom_hist = self.make_hist()

    def make_hist(self, index: Union[None, int] = None) -> np.ndarray:

        if index is None:
            # Take the nominal total weight
            weights = np.array(self.df[self.total_weight])
        else:
            # Divide with the nominal systematic weight and multiply with the varied one
            weights = np.array(
                (self.df[self.total_weight] / self.df[self.syst_weight])
                / self.df[f"{self.syst_weight}_var_{index}"]
            )

        return np.histogramdd(
            np.array(self.df[[*self.binning.keys()]]),
            bins=[bins for bins in self.binning.values()],
            weights=weights,
        )

    def visualize_variations(self, Nvar: int = 5):

        fig, ax = create_single_figure()

        for i in range(Nvar):

            v_hist = self.make_hist(index=i)

            bin_edges = [b for b in self.binning.values]
            x = np.linspace(0, 1, (bin_edges[0] - 1) * (bin_edges[1] - 1))

            plot_variation_on_axis(ax, x, v_hist / self.nom_hist, i)

        ax.set_ylabel("Template relative variation")
        ax.set_xlabel("Fitting variable")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
