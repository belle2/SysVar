from __future__ import annotations

from functools import cached_property

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from pandas import DataFrame

from sysvar.corrections import BaseCorrection
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
        syst_weight: Union[None, str] = None,
        correction: Union[None, BaseCorrection] = None,
        variator: Union[None, Variator] = None,
    ):
        columns = []
        if self._is_correct_binning(df.columns, binning):
            self.binning = binning
            columns.extend(list(binning.keys()))
        if self._is_existing_variable(total_weight, df.columns):
            self.total_weight = total_weight
            columns.append(total_weight)
        if syst_weight is not None:
            self._is_existing_variable(syst_weight, df.columns)
            columns.append(syst_weight)
        self.syst_weight = syst_weight
        # Make a deep copy only of the columns that are needed
        if correction is not None:
            columns.append(correction.dependant_variable)
        self.df = df[columns].copy(deep=True)

        self.correction = correction
        self.variator = variator
        self.Nvar = variator.Nvar if variator is not None else variator

    @property
    def cov_matrix(self) -> np.ndarray:
        return np.cov(self._get_absolute_variations())

    @property
    def corr_matrix(self) -> np.ndarray:
        return np.corrcoef(self._get_absolute_variations())

    @cached_property
    def eigen_decomposition(self) -> tuple:
        return np.linalg.eig(self.cov_matrix)

    @property
    def eigen_values(self) -> np.ndarray:
        return self.eigen_decomposition[0]

    @property
    def eigen_vectors(self) -> np.ndarray:
        return self.eigen_decomposition[1]

    @property
    def eigen_variations(self) -> np.ndarray:
        return self.eigen_vectors * np.sqrt(self.eigen_values)

    @property
    def Nbins(self) -> int:
        return np.prod([len(bins) - 1 for bins in self.binning.values()])

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

    @abstractmethod
    def make_hist(self, index: Union[None, int] = None) -> np.ndarray:
        pass

    def add_variations(self):

        # Initialize the nominal up and down variations
        self.df.loc[:, f"{self.syst_weight}_up"] = 1
        self.df.loc[:, f"{self.syst_weight}_down"] = 1

        # Initialize the variations
        self.df.loc[:, [f"{self.syst_weight}_var_{i}" for i in range(self.Nvar)]] = 1

        for i, (q, te) in enumerate(
            zip(self.correction.queries, self.correction.total_error)
        ):
            # Now add the variations of the corrections to the dataframe entries that
            # pass the cuts

            self.df.loc[self.df.eval(q), f"{self.syst_weight}_up"] = (
                self.df[self.syst_weight] + te
            )

            self.df.loc[self.df.eval(q), f"{self.syst_weight}_down"] = (
                self.df[self.syst_weight] - te
            )

            self.df.loc[
                self.df.eval(q),
                [f"{self.syst_weight}_var_{j}" for j in range(self.Nvar)],
            ] = self.variator.variations[:, i]

    def _get_absolute_variations(self):
        absolute_variations = np.empty((self.Nbins, self.Nvar))
        for i in range(self.Nvar):
            absolute_variations[:, i] = self.make_hist(i)[0].flatten()

        return absolute_variations


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
        syst_weight: Union[None, str] = None,
        correction: Union[None, BaseCorrection] = None,
        variator: Union[None, Variator] = None,
    ):
        super().__init__(df, binning, total_weight, syst_weight, correction, variator)

        self.nom_hist = self.make_hist()

    def make_hist(self, index: Union[None, int, str] = None) -> np.ndarray:

        if index is None:
            # Take the nominal total weight
            weights = np.array(self.df[self.total_weight])
        elif isinstance(index, str):
            if index == "MC":
                weights = np.square(self.df[self.total_weight])
            # FIXME This needs to be safely generalized
            elif (
                index in ["up", "down"]
                or index in [f"up{x}" for x in range(9)]
                or index in [f"down{x}" for x in range(9)]
            ):
                # PATCH
                # Now I'm replacing 0s with 1s to avoid NANs in the histogram
                weights = (
                    self.df[self.total_weight] / self.df[self.syst_weight].replace(0, 1)
                ) * (self.df["_".join((self.syst_weight, index))])
            else:
                raise NotImplementedError("only MC, up and down variations implemented")

        else:
            # Divide with the nominal systematic weight and multiply with the varied one
            weights = np.array(
                (self.df[self.total_weight] / self.df[self.syst_weight].replace(0, 1))
                * self.df[f"{self.syst_weight}_var_{index}"]
            )

        hist = np.histogramdd(
            np.array(self.df[[*self.binning.keys()]]),
            bins=[bins for bins in self.binning.values()],
            weights=weights,
        )
        return (hist[0].flatten(), np.linspace(0, 1, hist[0].flatten().shape[0] + 1))
