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


class CovarianceCalculator(SavableAttributesObject):

    def __init__(
        self,
        df: DataFrame,
        settings: dict,
        syst_effect: str,
        binning: dict,
        channels: int | List[str] = None,
        cov_input: np.ndarray | None = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.df = df
        self.settings = settings
        if isinstance(syst_effect, dict):
            # If the syst_effect is a dict, we need to extract the name
            self.syst_effect = syst_effect["name"]
        else:
            self.syst_effect = syst_effect

        # Pass the input systt effect in case we had a custom correction
        self.correction = create_correction_object(syst_effect, settings["MC_prod"])
        self.variator = Variator(self.correction, Nvar=settings["Nvar"])
        if cov_input is not None:
            self.variator.cov_matrix = cov_input

        self.binning = binning
        self.channels = channels
        self.verbose = verbose
        self.templates = self._create_varied_templates()

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
    def iterator(self) -> itertools.product:
        # Making this a propery to ensure that the iterations are always the same
        # This avoids inconsistencies when creating the templates and when saving them
        return itertools.product(self.channels, self.template_names)

    @property
    def cov(self) -> np.ndarray:
        for j, t in enumerate(self.template_names):
            # extract the template index as all of them are a big list
            template_index = j
            tmp_template = self.templates[template_index]
            # If this is the first templates initialize an emtpy cov matrix
            if j == 0:
                cov = np.zeros((tmp_template.Nbins, tmp_template.Nbins))
                # Now add the template cov matrix
            cov += tmp_template.cov_matrix
        return cov

    def _create_varied_templates(self):

        varied_templates = []
        # Extract column names from settings for readability
        reco_col = self.settings["reco_channel_id_column"]
        template_col = self.settings["template_id_column"]

        for template_name in self.template_names:

            # Apply the filter using .loc for better performance
            tmp_df = self.df.loc[
                (self.df[reco_col].isin(self.channels))
                & (self.df[template_col] == template_name)
            ]

            # Skip template create if the query yields an empty dataframe
            if len(tmp_df) < 1:
                if self.verbose:
                    logging.warn("Skipping template %s", str(template_name))
                    # PATCH FIX
                    # This just creates an empty dataframe so that TemplateND doesn't fail.
                # Maybe it's okay to keep doing this w/o a fix
                tmp_df = DataFrame(0, index=np.arange(1), columns=self.df.columns)

            template = self._get_template_child_class(self.binning)

            # TODO I don't like the switch between nominal and varied templates
            t = template(
                df=tmp_df,
                binning=self.binning,
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
            if self.verbose:
                if not len(tmp_df) < 1:
                    logging.info(
                        "Building %s for %s", str(type(t).__name__), str(template_name)
                    )

            if self.syst_effect is not None:
                t.add_variations()

            varied_templates.append(t)

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

    def save_covariance(self):

        # extract its topdir
        base_path = self.settings["covariance_matrices_dir"]

        # check if the covariance directory exists
        # if not path.exists(base_path):
        # if not create it
        makedirs(base_path, exist_ok=True)

        filename = "_".join(
            (
                "_".join([str(x) for x in self.channels]),
                "_".join(list(self.binning.keys())),
                "_".join(
                    [str(b) for bin_list in self.binning.values() for b in bin_list]
                ),
                self.syst_effect,
                "cov.npy",
            )
        )

        outpath = path.join(base_path, filename)
        np.save(outpath, self.cov)
        logging.info("Save covariance matrix at %s", str(outpath))
