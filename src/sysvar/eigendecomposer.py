from __future__ import annotations

from os import path, makedirs
import itertools
from functools import cached_property
from typing import List
from warnings import warn

from tqdm import tqdm

import numpy as np
from pandas import DataFrame

import uproot

from sysvar.corrections import create_correction_object
from sysvar.variations import Variator
from sysvar.templates import Template1D, TemplateND
from sysvar.visualize import EigenDecomposerVisualizer
from sysvar.channel_template_handler import ChannelTemplateHandler
from sysvar.utils import read_yaml


import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class EigenDecomposer(ChannelTemplateHandler):
    def __init__(
        self,
        df: DataFrame,
        settings: dict,
        syst_effect: str | dict | None = None,
        csv_path: str | None = None,
        title: str | None = None,
        verbose: bool = True,
        seed: int = 8311311,
    ):
        """
        Initialize an EigenDecomposer for systematic uncertainty analysis.

        Args:
            df (DataFrame): The input dataframe containing the data to be analyzed.
            settings (dict): Configuration settings containing analysis parameters.
            syst_effect (str | dict | None, optional): Systematic effect identifier for YAML-based corrections.
                Can be a string (systematic name) or dict (custom correction).
                Required if csv_path is not provided.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to True.
            csv_path (str | None, optional): Path to CSV file for CSV-based corrections.
                If provided, syst_effect is ignored.
            title (str | None, optional): Custom title for CSV-based corrections.
                If not provided, will use the CSV filename.

        Raises:
            ValueError: If neither syst_effect nor csv_path is provided, or if invalid types are passed.

        Examples:
            >>> # YAML-based correction
            >>> decomposer = EigenDecomposer(df, settings, syst_effect="track_eff")

            >>> # CSV-based correction
            >>> decomposer = EigenDecomposer(df, settings, csv_path="corrections/track_eff.csv")

        """

        super().__init__(df, settings, verbose)

        # Handle CSV-based corrections
        if csv_path is not None:
            self._syst_effect = (
                title
                if title is not None
                else str(path.basename(csv_path).replace(".csv", ""))
            )
            self.correction = create_correction_object(
                syst_effect=None, MC_prod=None, csv_path=csv_path, title=title
            )
        # Handle YAML-based corrections
        elif syst_effect is not None:
            warn(
                "Deprecation warning: YAML-based corrections from the Performance group are deprecated since MC16rd and will be removed in a future release. "
                "Please migrate to the CSV-based corrections. "
                "YAML corrections will remain available only for custom (user-provided) corrections, but future support is not guaranteed.",
                DeprecationWarning,
            )
            if isinstance(syst_effect, dict):
                self._syst_effect = syst_effect["name"]
            elif isinstance(syst_effect, str):
                self._syst_effect = syst_effect
            else:
                raise ValueError(
                    f"syst_effect must be a string or a dict but you passed {type(syst_effect)}"
                )

            self.correction = create_correction_object(syst_effect, settings["MC_prod"])
        else:
            raise ValueError(
                "Either syst_effect or csv_path must be provided. "
                "Use syst_effect for YAML-based corrections or csv_path for CSV-based corrections."
            )

        self.seed = seed
        self.variator = Variator(self.correction, Nvar=settings["Nvar"], seed=seed)
        self.N_important_dims = 0
        self.max_variations = None
        self.important_dims_indices = None

    @property
    def syst_weight(self) -> str:
        return self.settings["systematics"][self.syst_effect]["weight"]

    @property
    def prefices(self) -> str:
        return self.settings["systematics"][self.syst_effect]["prefices"]

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    @property
    def max_variations(self) -> int | None:
        return self._max_variations

    @max_variations.setter
    def max_variations(self, max_variations: int | None):
        self._max_variations = max_variations

    @cached_property
    def combined_variations(self) -> np.ndarray:
        return np.vstack(
            [x._get_absolute_variations() for x in self.templates.values()]
        )

    @property
    def cov(self) -> np.ndarray:
        return np.cov(self.combined_variations)

    # FIXME add a setter

    @property
    def corr(self) -> np.ndarray:
        return np.corrcoef(self.combined_variations)

    @cached_property
    def eigen_decomposition(self) -> tuple:
        return np.linalg.eig(self.cov)

    @property
    def eigen_values(self) -> np.ndarray:
        return self.eigen_decomposition[0]

    @property
    def eigen_vectors(self) -> np.ndarray:
        return self.eigen_decomposition[1]

    @property
    def eigen_variations(self) -> np.ndarray:
        return np.real(self.eigen_vectors * np.sqrt(self.eigen_values))

    @cached_property
    def max_differences(self) -> list:
        total_N = len(self.eigen_vectors)
        max_n = min(total_N, 100)  # only need the first 100
        n_start = 0  # Starting point for considering eigendirections when subtracting from the original covariance matrix
        running_cov = np.zeros_like(self.cov)
        # If the first eigenvalue is zero then the eigendirection is also zero
        # Completely skip this eigendirection but we still have to understand why this happens
        if self.eigen_values[0] == 0:
            max_n = max_n - 1  # Consider one less eigendirection in total
            n_start = 1  # skip the eigendirection that corresponds to 0 eigenvalue

        max_diffs = np.empty(max_n, dtype=float)

        logging.warning(
            "Only the first %d eigendirections will be considered to find "
            "the maximum number of eigenvariations.",
            max_n,
        )

        for i in tqdm(range(n_start, max_n), desc="Building partial covariances"):
            # grab the i-th eigen-variation (shape: [n_features,])
            vec = self.eigen_variations[:, i]

            # update the running sum
            running_cov += self.var2cov(vec)

            # compute the max difference
            diff = np.abs(np.real(self.cov - running_cov)).max()
            max_diffs[i] = diff

        return max_diffs

    @staticmethod
    def var2cov(mat) -> np.ndarray:
        return np.outer(mat, mat)

    def vary_templates(self):

        previous_reco_mode = None
        for (reco_mode, template_name), template in zip(
            self.iterator, self.templates.values()
        ):
            if reco_mode != previous_reco_mode:
                if self.verbose:
                    logging.info(
                        "########## Reco channel: %s ##########", str(reco_mode[1])
                    )

            # Update the systematic info of the templates
            template.syst_weight = self.syst_weight
            template.prefices = self.prefices
            template.correction = self.correction
            template.variator = self.variator
            template.Nvar = self.variator.Nvar

            # Keep only the necessary columns to avoid memory issues
            template.drop_unecessary_columns()
            if self.verbose:
                logging.info("Adding variations to %s template", str(template_name))
            template.add_variations()

            previous_reco_mode = reco_mode

    def find_important_eigendimension_indices(self, method: str = "max_differences"):

        if method == "max_differences":
            important_dims = self._calculate_max_differences()

        elif method == "trace":
            important_dims = self._calculate_trace()
        else:
            raise NotImplementedError(
                f"Available methods are: max_differences and trace"
            )

        # Increase number by one to get the last principal component
        self.N_important_dims = np.sum(important_dims) + 1
        # Limit the number of important dimensions if max_variations is set
        if self.max_variations is not None:
            self.N_important_dims = min(self.N_important_dims, self.max_variations)
        # Set the index of the last principal component to True
        important_dims[self.N_important_dims] = True
        # Collect the indices of the important dimensions including the last one
        self.important_dims_indices = important_dims
        logging.info(
            f"Found that %s eigendirections matter for %s per cent precision",
            self.N_important_dims,
            self.precision,
        )
        if (
            self.max_variations is not None
            and self.N_important_dims == self.max_variations
        ):
            logging.info(
                f"Keeping only the first %s eigendirections",
                self.max_variations,
            )

    def _calculate_max_differences(self):
        """
        Calculate the important dimensions based on maximum differences.

        This method identifies the important dimensions by comparing the maximum differences
        (scaled by the maximum value of the covariance matrix) to a specified precision threshold.

        Returns:
            list: A list of boolean values indicating the important dimensions. The length of the list
            corresponds to the number of maximum differences, with `True` indicating an important dimension
            and `False` indicating otherwise.

        Attributes:
            max_differences (array-like): The maximum differences for each dimension.
            cov (array-like): The covariance matrix.
            precision (float): The precision threshold for determining the important dimensions.

        Example:
            >>> self.max_differences = [0.2, 0.5, 0.8]
            >>> self.cov = np.array([[1, 0.1], [0.1, 1]])
            >>> self.precision = 0.1
            >>> self._calculate_max_differences()
            [True, True, True]
        """

        important_dims = (
            np.asarray(self.max_differences) / self.cov.max() > self.precision
        )
        return important_dims

    def _calculate_trace(self):
        """
        Calculate the trace and determine the important dimensions based on eigenvalues.

        This method computes the total trace by summing the square roots of the eigenvalues.
        It then iteratively computes the truncated trace by adding the square root of each eigenvalue
        and normalizes it by the total trace. When the normalized trace exceeds the specified precision,
        it identifies the number of important dimensions.

        Returns:
            list: A list of boolean values indicating the important dimensions. The length of the list
            corresponds to the number of eigenvalues, with `True` indicating an important dimension
            and `False` indicating otherwise.

        Attributes:
            eigen_values (array-like): The eigenvalues of the matrix.
            precision (float): The precision threshold for determining the important dimensions.

        Example:
            >>> self.eigen_values = [4, 1, 0.5]
            >>> self.precision = 0.95
            >>> self._calculate_trace()
            [True, True]
        """

        total_trace = np.sum(np.sqrt(self.eigen_values))
        truncated_trace = 0

        for i in range(1, len(self.eigen_values) + 1):
            truncated_trace += np.sqrt(self.eigen_values[i - 1])
            normalized_trace = truncated_trace / total_trace

            if normalized_trace / total_trace > self.precision:
                important_dims = [True] * i
                break
        return important_dims

    def _get_unrolled_variations(self):

        return self.eigen_variations[:, self.important_dims_indices].reshape(
            len(self.decay_modes),
            len(self.template_names),
            self.Nbins,
            self.N_important_dims,
        )

    def save_template_variations(self, filepath=None):

        # Override the global filepath in case we want to run on a batch with b2luigi
        filepath = self.settings["output_filepath"] if filepath is None else filepath
        with uproot.update(filepath) as newfile:
            logging.info(
                "Updating file with uproot: %s", self.settings["output_filepath"]
            )

            previous_tree = None

            index = 0
            for ((tree_i, tree), (ctgy_i, ctgy)), t in zip(
                self.enumerated_iterator, self.templates.values()
            ):

                if tree != previous_tree:
                    logging.info(50 * "#")
                    logging.info("########## Reco channel: %s ##########", str(tree[1]))
                    logging.info(50 * "#")

                nominal = t.make_hist()

                for n_var in range(self.N_important_dims):

                    var = self.eigen_variations[index : index + t.Nbins, n_var]

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_var{n_var+1}_up"
                    )
                    logging.info(
                        "Saving Up variation of MC template %s in TBranch: %s",
                        str(ctgy),
                        branch_name,
                    )

                    newfile[branch_name] = nominal[0] + var, nominal[1]

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_var{n_var+1}_down"
                    )
                    logging.info(
                        "Saving Down variation of %s in TBranch: %s",
                        str(ctgy),
                        branch_name,
                    )

                    newfile[branch_name] = nominal[0] - var, nominal[1]

                index += t.Nbins

                previous_tree = tree

    def save_toys(self, filepath=None):

        # Override the global filepath in case we want to run on a batch with b2luigi
        filepath = (
            self.settings["output_toy_filepath"] if filepath is None else filepath
        )
        with uproot.update(filepath) as newfile:
            logging.info(
                "Updating file with uproot: %s", self.settings["output_toy_filepath"]
            )

            previous_tree = None

            index = 0
            for ((tree_i, tree), (ctgy_i, ctgy)), t in zip(
                self.enumerated_iterator, self.templates.values()
            ):

                if tree != previous_tree:
                    logging.info(50 * "#")
                    logging.info("########## Reco channel: %s ##########", str(tree[1]))
                    logging.info(50 * "#")

                for n_var in range(self.settings["Nvar"]):

                    toy = t.make_hist(n_var)

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_toy{n_var+1}"
                    )
                    logging.info(
                        "Saving toy %d of MC template %s in TBranch: %s",
                        n_var,
                        str(ctgy),
                        branch_name,
                    )

                    newfile[branch_name] = toy[0], toy[1]

                previous_tree = tree

    def save_one_sigma_variations(self, filepath=None):

        # Override the global filepath in case we want to run on a batch with b2luigi
        filepath = (
            self.settings["output_extreme_filepath"] if filepath is None else filepath
        )
        with uproot.update(filepath) as newfile:
            logging.info(
                "Updating file with uproot: %s",
                self.settings["output_extreme_filepath"],
            )

            previous_tree = None

            index = 0
            for ((tree_i, tree), (ctgy_i, ctgy)), t in zip(
                self.enumerated_iterator, self.templates.values()
            ):

                if tree != previous_tree:
                    logging.info(50 * "#")
                    logging.info("########## Reco channel: %s ##########", str(tree[1]))
                    logging.info(50 * "#")

                for var in ["up", "down"]:

                    variation = t.make_hist(var)

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_{var}"
                    )
                    logging.info(
                        "Saving one sigma variation %s of MC template %s in TBranch: %s",
                        var,
                        str(ctgy),
                        branch_name,
                    )

                    newfile[branch_name] = variation[0], variation[1]

                    filedir = f"{tree[1]}/Data/{self.syst_effect}_{var}"

                    logging.info(
                        "Adding empty observed Data for region: %s in %s",
                        tree[1],
                        filedir,
                    )
                    # Save empty data now as we work only on Asimov
                    newfile[filedir] = np.array([0, 0, 0]), np.array([0, 1, 2, 3])

                previous_tree = tree

    def save_channel_covariance_matrices(self, filepath=None):

        # PATCH so this is the general output file of the analysis
        filepath = self.settings["output_filepath"] if filepath is None else filepath
        # extract its topdir
        base_path = path.dirname(filepath)
        # Loop over all the templates
        for j, tmp_template in enumerate(self.templates.values()):
            # If this is the first templates initialize an emtpy cov matrix
            if j == 0:
                cov = np.zeros((tmp_template.Nbins, tmp_template.Nbins))
                # Now add the template cov matrix
            cov += tmp_template.cov_matrix

            # Once we have looped over all the templates then save the cov matrix for this channel

        cov_path_dir = path.join(base_path, f"cov_matrices")
        # check if the covariance directory exists
        if not path.exists(cov_path_dir):
            # if not create it
            makedirs(cov_path_dir)

        outpath = path.join(cov_path_dir, f"{dm[1]}_{self.syst_effect}_cov.npy")
        logging.info("Save covariance matrix at %s ##########", str(outpath))
        np.save(outpath, cov)

    def plot_cov_diff(self, save: bool = False, filename: str = ""):

        self.visualizer = EigenDecomposerVisualizer(self)
        fig, ax = self.visualizer.plot_cov_diff(save=save, filename=filename)
        return fig, ax

    def plot_corr_matrix(self, save: bool = False, filename: str = ""):
        self.visualizer = EigenDecomposerVisualizer(self)
        fig, ax = self.visualizer.plot_corr_matrix(save=save, filename=filename)
        return fig, ax


class ExistingEigenVariationsSaver(ChannelTemplateHandler):

    def __init__(self, df: DataFrame, settings: dict, verbose: bool = True):
        super().__init__(df, settings, verbose)

    @property
    def weight(self) -> str:
        return self.settings["systematics"][self.syst_effect]["weight"]

    @property
    def prefices(self) -> str | list | None:
        return self.settings["systematics"][self.syst_effect]["prefices"]

    @property
    def Nvar(self) -> int:
        return self.settings["systematics"][self.syst_effect]["Nvar"]

    def save_existing_eigenvariations(self, filepath=None, verbose: bool = True):

        # Override the global filepath in case we want to run on a batch with b2luigi
        filepath = self.settings["output_filepath"] if filepath is None else filepath
        with uproot.update(filepath) as newfile:
            logging.info(
                "Updating file with uproot: %s", self.settings["output_filepath"]
            )
            previous_tree = None

            for ((tree_i, tree), (ctgy_i, ctgy)), t in zip(
                self.enumerated_iterator, self.templates.values()
            ):

                if tree != previous_tree:
                    logging.info(50 * "#")
                    logging.info("########## Reco channel: %s ##########", str(tree[1]))
                    logging.info(50 * "#")

                for n in range(self.Nvar):

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_var{n+1}_up"
                    )
                    if verbose:
                        logging.info(
                            "Saving Up variation of MC template %s in TBranch: %s",
                            str(ctgy),
                            branch_name,
                        )

                    t.syst_weight = self.weight
                    t.prefices = self.prefices
                    t.Nvar = self.Nvar

                    varied_histo = t.make_hist(f"up{n}")
                    newfile[branch_name] = varied_histo[0], varied_histo[1]

                    branch_name = self._get_TBranch_name(
                        tree[1], ctgy, f"{self.syst_effect}_var{n+1}_down"
                    )
                    if verbose:
                        logging.info(
                            "Saving Down variation of %s in TBranch: %s",
                            str(ctgy),
                            branch_name,
                        )
                    varied_histo = t.make_hist(f"down{n}")
                    newfile[branch_name] = varied_histo[0], varied_histo[1]

                previous_tree = tree
