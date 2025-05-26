from __future__ import annotations

from os import path, makedirs
import itertools
from functools import cached_property
from typing import List

from tqdm import tqdm

import numpy as np
from pandas import DataFrame

import uproot

from sysvar.utils import SavableAttributesObject
from sysvar.templates import Template1D, TemplateND

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class ChannelTemplateHandler(SavableAttributesObject):
    def __init__(self, df: DataFrame, settings: dict, verbose: bool = True):

        super().__init__()

        self.df = df
        self.settings = settings
        # Make this None to allow the methods to be called from this base class and the EigeDecomposerr
        self._syst_effect = None
        self.verbose = verbose

    @property
    def syst_effect(self) -> str:
        return self._syst_effect

    @syst_effect.setter
    def syst_effect(self, value):
        if not isinstance(value, str):
            raise TypeError("syst_effect must be a string")
        self._syst_effect = value

    @cached_property
    def templates(self):
        """Creates templates based on the provided DataFrame and settings.

        Returns:
            list: A list of created templates.

        Example:
            >>> templates = self.create_templates()
            >>> len(templates)
            5
        """
        return self.create_templates()

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
        return np.vstack([x.make_hist()[0] for x in self.templates.values()])

    def create_templates(self):

        previous_reco_mode = None
        templates = {}
        # Extract column names from settings for readability
        reco_col = self.settings["reco_channel_id_column"]
        template_col = self.settings["template_id_column"]

        for reco_mode, template_name in self.iterator:

            if reco_mode != previous_reco_mode:
                if self.verbose:
                    logging.info(
                        "########## Reco channel: %s ##########", str(reco_mode[1])
                    )

            # Apply the filter using .loc for better performance
            tmp_df = self.df.loc[
                (self.df[reco_col].isin(reco_mode[0]))
                & (self.df[template_col] == template_name)
            ]

            empty_dataframe_flag = False
            # Skip template create if the query yields an empty dataframe
            if len(tmp_df) < 1:
                empty_dataframe_flag = True
                # PATCH FIX
                # This just creates an empty dataframe so that TemplateND doesn't fail.
                # Maybe it's okay to keep doing this w/o a fix
                tmp_df = DataFrame(0, index=np.arange(1), columns=self.df.columns)

            template = self._get_template_child_class(
                self.settings["bins"][reco_mode[1]]
            )

            if self.verbose:
                if empty_dataframe_flag:
                    logging.warn(
                        "Skipping template %s. %s events",
                        str(template_name),
                        str(len(tmp_df) - 1),
                    )
                else:
                    logging.info(
                        "Building %s for %s from %s events",
                        template.__name__,
                        str(template_name),
                        str(len(tmp_df) - 1),
                    )

            t = template(
                df=tmp_df,
                binning=self.settings["bins"][reco_mode[1]],
                total_weight=self.settings["total_weight"],
                verbose=self.verbose,
            )

            templates[f"{reco_mode[1]}_{template_name}"] = t

            previous_reco_mode = reco_mode

        return templates

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

    @staticmethod
    def _get_TBranch_name(*args):
        return "/".join(args)

    def save_nominal_templates(self, filepath=None, data=None):

        # FIX there needs to be a check here to not recreate the file if it already exists or has nominal templates.
        # Now the user needs to be careful to not mess up the file and lose previously written nominals
        # Override the global filepath in case we want to run on a batch with b2luigi

        if self.syst_effect is not None:
            raise ValueError(
                "You are trying to save nominal templates but have defined a systematic effect for the ChannelTemplateHandler at the same time. You probably have done this explicitly by setting the property `syst_effect`. Please consider setting the property to None (or let it default to None)"
            )

        filepath = self.settings["output_filepath"] if filepath is None else filepath
        with uproot.recreate(filepath, compression=None) as newfile:
            logging.info("Recreate file with uproot: %s", filepath)

            previous_tree = None

            for (tree, ctgy), t in zip(self.iterator, self.templates.values()):

                if tree != previous_tree:
                    if self.verbose:
                        logging.info(50 * "#")
                        logging.info(
                            "########## Reco channel: %s ##########", str(tree[1])
                        )
                        logging.info(50 * "#")

                branch_name = self._get_TBranch_name(tree[1], ctgy, "Nominal")
                if self.verbose:
                    logging.info(
                        "Saving Nominal MC template %s in TBranch: %s",
                        str(ctgy),
                        branch_name,
                    )

                newfile[branch_name] = t.make_hist()
                previous_tree = tree

            if self.verbose:
                logging.info(50 * "#")
                logging.info("########## Observed data ##########")
                logging.info(50 * "#")
            for tree in self.decay_modes:
                filedir = f"{tree[1]}/Data/Nominal"
                if data is None:
                    if self.verbose:
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
                    if self.verbose:
                        logging.info(
                            "Adding observed Data region : %s in %s", tree[1], filedir
                        )
                    newfile[filedir] = hist[0].flatten(), np.linspace(
                        0, 1, hist[0].flatten().shape[0] + 1
                    )
