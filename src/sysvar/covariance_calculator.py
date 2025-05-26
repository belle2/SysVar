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
from sysvar.utils import SavableAttributesObject
from sysvar.eigendecomposer import EigenDecomposer

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class CovarianceCalculator(EigenDecomposer):

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

        super().__init__(df, settings, syst_effect, verbose)

        if cov_input is not None:
            self.variator.cov_matrix = cov_input
            self.variator.variations = self.variator.get_correction_variations()

        self.binning = binning
        self.channels = channels

    @property
    def cov(self) -> np.ndarray:
        for j, tmp_template in enumerate(self.templates.values()):
            # If this is the first templates initialize an emtpy cov matrix
            if j == 0:
                cov = np.zeros((tmp_template.Nbins, tmp_template.Nbins))
                # Now add the template cov matrix
            cov += tmp_template.cov_matrix
            return cov

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
