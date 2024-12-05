from __future__ import annotations

from functools import cached_property

from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame, concat

from sysvar.corrections import *
from sysvar.variations import Variator
from sysvar.visualize import TemplateVisualizer
from sysvar.utils import SavableAttributesObject


class NotADictError(Exception):
    pass


class NotCorrectVariableError(Exception):
    pass


class NotIncreasingBinning(Exception):
    pass


class Template(ABC, SavableAttributesObject):
    def __init__(
        self,
        df: DataFrame,
        binning: dict,
        total_weight: str,
        syst_weight: None | str = None,
        prefices: None | str | list = None,
        correction: None | BaseCorrection = None,
        variator: None | Variator = None,
    ):
        super().__init__()
        if self._is_correct_binning(df.columns, binning):
            self.binning = binning
        if self._is_existing_variable(total_weight, df.columns):
            self.total_weight = total_weight

        # TODO have a method that build the column name e.g. from prefix and syst_weight and
        # add a check of is_existing_variable
        self.syst_weight = syst_weight
        self.prefices = prefices

        # Here the deep copy ensures that we're not affecting the original dataframe
        self.correction = correction
        self.df = df[self._collect_columns_names()].copy(deep=True)
        # TODO Make a deep copy only of the columns that are needed
        # But for large dataframes we don't need to copy all these GB. only a handful of columns are necessary.
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
    def make_hist(self, index: None | int = None) -> np.ndarray:
        pass

    def _collect_columns_names(self):
        """
        Collects and returns a list of column names based on the configuration of the object.

        The method generates column names by combining a fixed set of base columns
        (such as "channel" and "fit_ctgy") with dynamic elements like binning keys,
        system weights, and variables specific to the type of correction applied to
        the data. The correction could be of type `Correction1D`, `CorrectionBF`, or `CorrectionPID`,
        and each type defines a different set of variables. Column names are prefixed
        with one or more `prefices` defined for the object.

        Returns:
            list: A list of strings representing the collected column names.

        Raises:
            TypeError: If the type of `self.correction` is not recognized.

        Notes:
            - If `self.prefices` is not a list, it will be converted into one.
            - For each prefix in `self.prefices`, a column name is generated for each
              variable in the correction, using the method `_build_column_name`.
            - The method constructs column names by appending variables to a set of
              base columns such as channel, fit_ctgy, and binning keys.
        """

        # Initialize the base columns.
        # TODO the channel and fit_ctgy should not be hardcoded here
        columns = ["channel", "template", self.total_weight, *list(self.binning.keys())]

        # Collect the prefices
        prefices = (
            [x for x in self.prefices]
            if isinstance(self.prefices, list)
            else [self.prefices]
        )

        # Collect the important variables based on the type of correction
        if isinstance(self.correction, Correction1D):
            variables = [
                self.syst_weight,
                self.correction.dependant_variable,
                *list(self.correction.info["extra_cuts"].keys()),
            ]
        elif isinstance(self.correction, CorrectionBF):
            variables = [self.syst_weight, self.correction.dependant_variable]
        elif isinstance(self.correction, Correction2D):
            variables = [
                self.syst_weight,
                self.correction.dependant_variable_1,
                self.correction.dependant_variable_2,
                *list(self.correction.info["extra_cuts"].keys()),
            ]
        elif isinstance(self.correction, CorrectionPID):
            variables = [
                self.syst_weight,
                self.correction.p,
                self.correction.theta,
                self.correction.PDG,
                self.correction.mcPDG,
            ]
        elif self.correction is None:
            variables = []

        # Construct the column names
        for prefix in prefices:
            for var in variables:
                columns.append(self.correction._build_column_name(prefix, var))

        return columns

    def add_variations(self):

        if isinstance(self.prefices, str) or self.prefices is None:
            # If we are dealing with a single column, just build the name and add the variations
            weightname = self.correction._build_column_name(
                self.prefices, self.syst_weight
            )
            self._initialize_variations(weightname)
            self._add_variations_to_df(weightname, self.prefices)
        elif isinstance(self.prefices, list):
            # For multuple columns we need to loop over all the prefices.
            # Assumes that all the weights have the same name
            for prefix in self.prefices:
                # Now build the weightname
                weightname = self.correction._build_column_name(
                    prefix, self.syst_weight
                )

                # And initialize and add all the variations
                self._initialize_variations(weightname)
                self._add_variations_to_df(weightname, prefix)

        else:
            raise ValueError(
                f"The prefices must be str or a list of str. You passed {type(self.prefices)}"
            )

    def _initialize_variations(self, weightname: str):
        # Initialize the nominal up and down variations
        self.df[f"{weightname}_up"] = 1.0
        self.df[f"{weightname}_down"] = 1.0

        # Initialize the variations
        # Create a new dataframe ana concatenate it to avoid PerformanceWarning
        variation_columns = [f"{weightname}_var_{i}" for i in range(self.Nvar)]
        variations = DataFrame(
            data=np.ones((len(self.df), len(variation_columns))),
            columns=variation_columns,
        )
        # The index needs to be reseted, otherwise pandas will create extra rows
        # Here we overwrite the Templates dataframe
        # This is okay since this is only a copy of the original dataframe passed by the user.
        self.df = concat([self.df.reset_index(drop=True), variations], axis=1)

    def _add_variations_to_df(self, weightname: str, prefix: None | str = None):

        # Build the queries based on the prefix
        queries = self.correction.build_queries(prefix)

        # Create a list to store the mask results for each query
        masks = [self.df.query(q).index for q in queries]

        # Update DataFrame using precomputed masks
        for i, (mask, te) in enumerate(zip(masks, self.correction.total_error)):

            # Perform vectorized updates with the precomputed mask
            self.df.loc[mask, f"{weightname}_up"] = self.df.loc[mask, weightname] + te
            self.df.loc[mask, f"{weightname}_down"] = self.df.loc[mask, weightname] - te

            # Assign variations in a vectorized way using column names
            variation_columns = [f"{weightname}_var_{j}" for j in range(self.Nvar)]
            self.df.loc[mask, variation_columns] = self.variator.variations[:, i]

    def _combine_variations(self):

        # Commenting out, I think I don't need this at all
        # This could only be useful for the FF stuff but there we don't
        # calculate eigenvariations ourselves

        # self.df.loc[:, "combination_up"] = self.df[
        #    [f"{x}_up" for x in self.syst_weight]
        # ].prod(axis=1)
        # self.df.loc[:, "combination_down"] = self.df[
        #    [f"{x}_down" for x in self.syst_weight]
        # ].prod(axis=1)

        for j in range(self.Nvar):
            # Collect columns to be multiplied. These are all all the corrected particles
            columns_to_multiply = [
                f"{x}_{y}_var_{j}" for x, y in self.syst_weight.items()
            ]

            # Multiply the selected columns. Essentially multiplying the corrections to
            # to get only one back
            product_column = self.df[columns_to_multiply].prod(axis=1)

            # Concatenate the product column to the DataFrame
            self.df = concat(
                [self.df, product_column.rename(f"combination_var_{j}")], axis=1
            )

    def _get_absolute_variations(self):
        absolute_variations = np.empty((self.Nbins, self.Nvar))
        for i in range(self.Nvar):
            absolute_variations[:, i] = self.make_hist(i)[0].flatten()

        return absolute_variations

    def collect_weights(self, index):
        """Collects weights based on the provided index, handling different variations.

        Args:
            index: Specifies the type of weight to collect. Can be `None`, a string such as "MC",
                   "up", "down", or variations like "up0", "up1", ..., "up8", "down0", "down1", ..., "down8",
                   or an integer.

        Returns:
            numpy.ndarray: An array of weights based on the specified index.

        Raises:
            NotImplementedError: If the provided index is not supported.

        Notes:
            - If `index` is `None`, returns the nominal total weight.
            - If `index` is "MC", returns the square of the total weight.
            - For "up" and "down" variations, computes the weights with specific adjustments.
            - Handles both string and dictionary types for `self.syst_weight`.

        Example:
            >>> self.collect_weights(None)
            array([...])

            >>> self.collect_weights("MC")
            array([...])

            >>> self.collect_weights("up1")
            array([...])
        """
        if index is None:
            # Take the nominal total weight
            weights = np.array(self.df[self.total_weight])
        elif isinstance(index, str):
            if index == "MC":
                weights = np.square(self.df[self.total_weight])
            # FIXME This needs to be safely generalized
            # This is aligned for Felix's tuples now
            elif (
                index in ["up", "down"]
                or index in [f"up{x}" for x in range(9)]
                or index in [f"down{x}" for x in range(9)]
            ):
                # PATCH
                # Now I'm replacing 0s with 1s to avoid NANs in the histogram
                if isinstance(self.prefices, str):
                    # PATCH
                    weightname = "_".join([self.prefices, self.syst_weight])
                    weights = (
                        self.df[self.total_weight] / self.df[weightname].replace(0, 1)
                    ) * (self.df["_".join((weightname, index))])
                else:
                    raise NotImplementedError(
                        "Only one prefix at a time currently! Revisit this for HID and pi0s"
                    )

            else:
                raise NotImplementedError("only MC, up and down variations implemented")
        else:
            # Divide with the nominal systematic weight and multiply with the varied one
            if isinstance(self.prefices, str):
                weightname = "_".join([self.prefices, self.syst_weight])
                weights = np.array(
                    (self.df[self.total_weight] / self.df[weightname].replace(0, 1))
                    * self.df[f"{weightname}_var_{index}"]
                )

            elif self.prefices is None:
                weightname = self.syst_weight
                weights = np.array(
                    (self.df[self.total_weight] / self.df[weightname].replace(0, 1))
                    * self.df[f"{weightname}_var_{index}"]
                )

            elif isinstance(self.prefices, list):
                # If we have multiple prefices we need to to create all the column names first
                weightnames = [
                    "_".join([prefix, self.syst_weight]) for prefix in self.prefices
                ]
                # We divide with the product of all the nominal weights
                # and multiply with the product of all of those which we added the suffix to
                weights = np.array(
                    self.df[self.total_weight]
                    / self.df[weightnames].replace(0, 1).prod(axis=1)
                    * self.df[[f"{w}_var_{index}" for w in weightnames]].prod(axis=1)
                )

            else:
                raise NotImplementedError(
                    "Only one prefix at a time currently! Revisit this for HID and pi0s"
                )

        return weights

    def plot_systematic_overview(self, save: bool = False, filename: str = ""):

        self.visualizer = TemplateVisualizer(self)
        self.visualizer.plot_systematic_overview(save=save, filename=filename)

    def plot_relative_variations_in_grid(self, save: bool = False, filename: str = ""):

        self.visualizer = TemplateVisualizer(self)
        self.visualizer.plot_relative_variations_in_grid(save=save, filename=filename)

    def plot_up_and_down_variations(self, save: bool = False, filename: str = ""):

        self.visualizer = TemplateVisualizer(self)
        self.visualizer.plot_up_and_down_variations(save=save, filename=filename)


class Template1D(Template):
    def __init__(
        self,
        df: DataFrame,
        binning: dict,
        total_weight: str,
        syst_weight: str = None,
        prefices: str | list = None,
        correction: None | BaseCorrection = None,
        variator: None | Variator = None,
    ):
        super().__init__(
            df, binning, total_weight, syst_weight, prefices, correction, variator
        )

        self.nom_hist = self.make_hist()

    def make_hist(self, index: None | int | str = None) -> np.ndarray:

        weights = self.collect_weights(index)

        hist = np.histogram(
            np.array(self.df[list(self.binning.keys())[0]]),
            bins=list(*self.binning.values()),
            weights=weights,
        )
        return hist[0].flatten(), hist[1].flatten()


class TemplateND(Template):
    def __init__(
        self,
        df: DataFrame,
        binning: dict,
        total_weight: str,
        syst_weight: None | str = None,
        prefices: str | list = None,
        correction: None | BaseCorrection = None,
        variator: None | Variator = None,
    ):
        super().__init__(
            df, binning, total_weight, syst_weight, prefices, correction, variator
        )

        self.nom_hist = self.make_hist()

    def make_hist(self, index: None | int | str = None) -> np.ndarray:

        weights = self.collect_weights(index)

        hist = np.histogramdd(
            np.array(self.df[[*self.binning.keys()]]),
            bins=[bins for bins in self.binning.values()],
            weights=weights,
        )
        return (hist[0].flatten(), np.linspace(0, 1, hist[0].flatten().shape[0] + 1))
