from __future__ import annotations

from os import path, makedirs
import itertools
from functools import cached_property
from typing import List

from tqdm import tqdm

import numpy as np
from pandas import DataFrame

import uproot

from sysvar.corrections import create_correction_object
from sysvar.variations import Variator
from sysvar.templates import Template1D, TemplateND
from sysvar.visualize import EigenDecomposerVisualizer
from sysvar.utils import SavableAttributesObject

from sysvar.utils import read_yaml, SavableAttributesObject

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class EigenDecomposer(SavableAttributesObject):
    def __init__(self, df: DataFrame, settings: dict, syst_effect: str):

        super().__init__()

        self.df = df
        self.settings = settings
        if isinstance(syst_effect, dict):
            self.syst_effect = syst_effect["name"]
        else:
            self.syst_effect = syst_effect

        if syst_effect is not None:
            self.correction = create_correction_object(syst_effect, settings["MC_prod"])
            self.variator = Variator(self.correction, Nvar=settings["Nvar"])
            self.N_important_dims = 0
            self.important_dims_indices = None
        else:
            # This is useful for saving nominal templates
            self.correction = None
            self.variator = None
        self.templates = self._create_varied_templates()

    @property
    def decay_modes(self) -> list:
        return [
            (self.settings["reco_channels"][reco_channel], reco_channel)
            for reco_channel in self._determine_reco_channels()
        ]

    @property
    def _included_channels(self) -> str | List[str]:
        """Retrieves the list of included reconstruction channels for the current systematic effect.

        Returns:
            str | List[str]: A list of included channels or a single channel as a string.

        Notes:
            - If no channels are explicitly included, returns all available reconstruction channels.

        Example:
            >>> self._included_channels
            ['channel1', 'channel2']
        """
        if (
            self.settings["systematics"][self.syst_effect]["reco_channels"]["include"]
            is None
        ):
            return list(self.settings["reco_channels"].keys())
        else:
            return self.settings["systematics"][self.syst_effect]["reco_channels"][
                "include"
            ]

    @property
    def _excluded_channels(self) -> None | str | List[str]:
        """Retrieves the list of excluded reconstruction channels for the current systematic effect.

        Returns:
            None | str | List[str]: A list of excluded channels, a single channel as a string, or an empty list if none are excluded.

        Example:
            >>> self._excluded_channels
            ['channel3', 'channel4']
        """
        if (
            self.settings["systematics"][self.syst_effect]["reco_channels"]["exclude"]
            is None
        ):
            return []
        else:
            return self.settings["systematics"][self.syst_effect]["reco_channels"][
                "exclude"
            ]

    def _determine_reco_channels(self) -> List[str]:
        """Determines the final list of reconstruction channels to be used, based on inclusion and exclusion criteria.

        Returns:
            List[str]: A list of reconstruction channel names that are included and not excluded.

        Example:
            >>> self._determine_reco_channels()
            ['channel1', 'channel2']
        """
        if self.syst_effect is None:
            reco_channels = self.settings["reco_channels"].keys()
        else:
            reco_channels = [
                reco_channel_name
                for reco_channel_name in self.settings["reco_channels"].keys()
                if reco_channel_name in self._included_channels
                and reco_channel_name not in self._excluded_channels
            ]
        return reco_channels

    @property
    def template_names(self) -> list:

        if (
            self.syst_effect is not None
            and self.settings["systematics"][self.syst_effect]["templates"] is not None
        ):
            return [
                template_name
                for template_name in self.settings["systematics"][self.syst_effect][
                    "templates"
                ]
            ]
        else:
            return [template_name for template_name in self.settings["templates"]]

    @property
    def Nbins(self) -> int:
        return np.prod([len(bins) - 1 for bins in self.settings["bins"].values()])

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    @property
    def iterator(self) -> itertools.product:
        # Making this a propery to ensure that the iterations are always the same
        # This avoids inconsistencies when creating the templates and when saving them
        return itertools.product(self.decay_modes, self.template_names)

    @property
    def enumerated_iterator(self) -> itertools.product:
        # Making this a propery to ensure that the iterations are always the same
        # This avoids inconsistencies when creating the templates and when saving them
        return itertools.product(
            enumerate(self.decay_modes), enumerate(self.template_names)
        )

    @cached_property
    def nominal_hist(self) -> np.ndarray:
        return np.vstack([x.make_hist()[0] for x in self.templates])

    @cached_property
    def combined_variations(self) -> np.ndarray:
        return np.vstack([x._get_absolute_variations() for x in self.templates])

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

        total_N_vectors = len(self.eigen_vectors)
        max_differences = []
        logging.warn(
            "Only the first 100 eigendirections will be considered to find the maximum number of eigenvariations. This is an arbitrary choice as it's highly unlikely that an analysis will use > 100 nuisance parameters for only one systematic source."
        )
        for n_vectors in tqdm(range(total_N_vectors)):

            dimension_subset = self.eigen_variations[:, :n_vectors].T
            subset_covariances = [self.var2cov(x) for x in dimension_subset]

            max_differences.append(
                np.abs(np.real(self.cov - np.sum(subset_covariances, axis=0))).max()
            )

            # Calculate only the first 50 eigendirections to save time
            # It's highly unlikely that an analysis will use > 50 nuisance parameters
            # for one systematic only
            if n_vectors > 99:
                break

        return max_differences

    @staticmethod
    def var2cov(mat) -> np.ndarray:
        return np.outer(mat, mat)

    def _create_varied_templates(self):

        previous_reco_mode = None
        varied_templates = []
        # Extract column names from settings for readability
        reco_col = self.settings["reco_channel_id_column"]
        template_col = self.settings["template_id_column"]

        for reco_mode, template_name in self.iterator:

            if reco_mode != previous_reco_mode:
                logging.info(
                    "########## Reco channel: %s ##########", str(reco_mode[1])
                )

            # Apply the filter using .loc for better performance
            tmp_df = self.df.loc[
                (self.df[reco_col].isin(reco_mode[0]))
                & (self.df[template_col] == template_name)
            ]

            # Skip template create if the query yields an empty dataframe
            if len(tmp_df) < 1:
                logging.warn("Skipping template %s", str(template_name))
                # PATCH FIX
                # This just creates an empty dataframe so that TemplateND doesn't fail.
                # Maybe it's okay to keep doing this w/o a fix
                tmp_df = DataFrame(0, index=np.arange(1), columns=self.df.columns)

            template = self._get_template_child_class(
                self.settings["bins"][reco_mode[1]]
            )

            # TODO I don't like the switch between nominal and varied templates
            t = template(
                df=tmp_df,
                binning=self.settings["bins"][reco_mode[1]],
                total_weight=self.settings["total_weight"],
                syst_weight=(
                    self.settings["systematics"][self.syst_effect]["weight"]
                    if self.syst_effect is not None
                    else None
                ),
                prefices=(
                    self.settings["systematics"][self.syst_effect]["prefices"]
                    if self.syst_effect is not None
                    else None
                ),
                correction=self.correction,
                variator=self.variator,
            )

            # FIX this defaults to TemplateND logging. Make it more generic
            if not len(tmp_df) < 1:
                logging.info(
                    "Building %s for %s", str(type(t).__name__), str(template_name)
                )

            if self.syst_effect is not None:
                t.add_variations()

            varied_templates.append(t)

            previous_reco_mode = reco_mode

        return varied_templates

    @staticmethod
    def _get_template_child_class(binning):
        """Determines the appropriate template class based on the dimensionality of the binning.

        Args:
            binning (dict): A dictionary representing the binning configuration.

        Returns:
            class: The appropriate template class (`Template1D` or `TemplateND`).

        Raises:
            NotImplementedError: If the binning dimensionality is not 1D or 2D.

        Example:
            >>> binning = {'x': [0, 1, 2, 3]}
            >>> template_class = _get_template_child_class(binning)
            >>> template_class
            <class 'Template1D'>

            >>> binning = {'x': [0, 1, 2, 3], 'y': [0, 1, 2]}
            >>> template_class = _get_template_child_class(binning)
            >>> template_class
            <class 'TemplateND'>

            >>> binning = {'x': [0, 1, 2, 3], 'y': [0, 1, 2], 'z': [0, 1]}
            >>> template_class = _get_template_child_class(binning)
            Traceback (most recent call last):
                ...
            NotImplementedError: Only 1D and 2D histograms are implemented so far. Please check the binning of your reconstruction channels.
        """
        if len(binning.keys()) == 1:
            template = Template1D
        elif len(binning.keys()) > 1:
            template = TemplateND
        else:
            raise NotImplementedError(
                "Only 1D and ND histograms are implemented so far. Please check the binning of your reconstruction channels."
            )

        return template

    def find_important_eigendimension_indices(self, method: str = "max_differences"):

        if method == "max_differences":
            important_dims = self._calculate_max_differences()

        elif method == "trace":
            important_dims = self._calculate_trace()
        else:
            raise NotImplementedError(
                f"Available methods are: max_differences and trace"
            )

        self.N_important_dims = np.sum(important_dims)
        self.important_dims_indices = important_dims
        logging.info(
            f"Found that %s eigendirections matter for %s per cent precision",
            self.N_important_dims,
            self.precision,
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

    @staticmethod
    def _get_TBranch_name(*args):
        return "/".join(args)

    def save_nominal_templates(self, filepath=None, data=None):

        # FIX there needs to be a check here to not recreate the file if it already exists or has nominal templates.
        # Now the user needs to be careful to not mess up the file and lose previously written nominals
        # Override the global filepath in case we want to run on a batch with b2luigi
        filepath = self.settings["output_filepath"] if filepath is None else filepath
        with uproot.recreate(filepath, compression=None) as newfile:
            logging.info("Recreate file with uproot: %s", filepath)

            previous_tree = None

            for (tree, ctgy), t in zip(self.iterator, self.templates):

                if tree != previous_tree:
                    logging.info(50 * "#")
                    logging.info("########## Reco channel: %s ##########", str(tree[1]))
                    logging.info(50 * "#")

                branch_name = self._get_TBranch_name(tree[1], ctgy, "Nominal")
                logging.info(
                    "Saving Nominal MC template %s in TBranch: %s",
                    str(ctgy),
                    branch_name,
                )

                newfile[branch_name] = t.make_hist()
                previous_tree = tree

            logging.info(50 * "#")
            logging.info("########## Observed data ##########")
            logging.info(50 * "#")
            for tree in self.decay_modes:
                filedir = f"{tree[1]}/Data/Nominal"
                if data is None:
                    logging.info(
                        "Adding empty observed Data for region: %s in %s",
                        tree[1],
                        filedir,
                    )
                    # Save empty data now as we work only on Asimov
                    newfile[filedir] = np.array([0, 0, 0]), np.array([0, 1, 2, 3])
                else:
                    if not isinstance(data, DataFrame):
                        raise TypeError("data must be a pandas DataFrame")

                    reco_col = self.settings["reco_channel_id_column"]
                    binning = self.settings["bins"][tree[1]]

                    hist = np.histogramdd(
                        np.array(
                            data.query(f"{reco_col} in {tree[0]}")[[*binning.keys()]]
                        ),
                        bins=[bins for bins in binning.values()],
                    )
                    logging.info(
                        "Adding observed Data region : %s in %s", tree[1], filedir
                    )
                    newfile[filedir] = hist[0].flatten(), np.linspace(
                        0, 1, hist[0].flatten().shape[0] + 1
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
                self.enumerated_iterator, self.templates
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
                self.enumerated_iterator, self.templates
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
                self.enumerated_iterator, self.templates
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
        # Loop over all the channels
        for i, dm in enumerate(self.decay_modes):
            logging.info("########## Reco channel: %s ##########", str(dm[1]))
            # Loop over all the templates
            for j, t in enumerate(self.template_names):
                # extract the template index as all of them are a big list
                template_index = len(self.template_names) * i + (j)
                tmp_template = self.templates[template_index]
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
        self.visualizer.plot_cov_diff(save=save, filename=filename)

    def plot_corr_matrix(self, save: bool = False, filename: str = ""):
        self.visualizer = EigenDecomposerVisualizer(self)
        self.visualizer.plot_corr_matrix(save=save, filename=filename)
