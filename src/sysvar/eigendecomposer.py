from functools import cached_property

from tqdm import tqdm

import numpy as np
from pandas import DataFrame

import uproot

from sysvar.corrections import Correction, BFCorrection
from sysvar.variations import Variator
from sysvar.templates import Template2D

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class EigenDecomposer:
    def __init__(self, df: DataFrame, settings: dict, syst_effect: str):

        self.df = df
        self.settings = settings
        self.syst_effect = syst_effect

        if settings["systematics"][syst_effect]["BF"]:
            correction_type = BFCorrection
        else:
            correction_type = Correction
        self.correction = correction_type(
            dependant_variable=settings["systematics"][syst_effect]["var"],
            systematic=syst_effect,
            MC_production=settings["MC_prod"],
        )
        self.variator = Variator(self.correction, Nvar=settings["Nvar"])
        self.varied_templates = self._get_varied_templates()
        self.N_important_dims = 0
        self.important_dims_indices = None

    @cached_property
    def decay_modes(self) -> list:
        return [
            (reco_mode, treename)
            for reco_mode, treename in zip(
                self.settings["systematics"][self.syst_effect]["regions"],
                self.settings["systematics"][self.syst_effect]["tree_names"],
            )
        ]

    @cached_property
    def fit_ctgies(self) -> list:
        return [fit_ctgy for fit_ctgy in self.settings["fit_ctgies"]]

    @property
    def Nbins(self) -> int:
        return np.prod([len(bins) - 1 for bins in self.settings["bins"].values()])

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    def _get_varied_templates(self):

        varied_templates = []
        for reco_mode in self.decay_modes:
            logging.info("Building in reco region %s", str(reco_mode[1]))
            reco_df = self.df.query(
                f"{self.settings['region_id_column']} in @reco_mode[0]"
            )
            for fit_ctgy in self.fit_ctgies:
                logging.info("Building template %s", str(fit_ctgy))

                tmp_df = reco_df.query(
                    f"{self.settings['ctgy_id_column']} == '{fit_ctgy}'"
                )
                # Skip template create if the query yields an empty dataframe
                if len(tmp_df) < 1:
                    logging.warn("Skipping template %s", str(fit_ctgy))
                    # PATCH FIX
                    # This just creates an empty dataframe so that Template2D doesn't fail.
                    # Maybe it's okay to keep doing this w/o a fix
                    tmp_df = DataFrame(0, index=np.arange(1), columns=self.df.columns)

                t = Template2D(
                    tmp_df,
                    self.settings["bins"],
                    self.settings["weight"],
                    self.settings["systematics"][self.syst_effect]["weight"],
                    self.correction,
                    self.variator,
                )

                t.add_variations()

                varied_templates.append(t)

        return varied_templates

    @cached_property
    def nominal_hist(self) -> np.ndarray:
        return np.vstack([x.make_hist()[0] for x in self.varied_templates])

    @cached_property
    def combined_variations(self) -> np.ndarray:
        return np.vstack([x._get_absolute_variations() for x in self.varied_templates])

    @property
    def cov(self) -> np.ndarray:
        return np.cov(self.combined_variations)

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
        return self.eigen_vectors * np.sqrt(self.eigen_values)

    @cached_property
    def max_differences(self) -> list:

        total_N_vectors = len(self.eigen_vectors)
        max_differences = []
        for n_vectors in tqdm(range(total_N_vectors)):

            dimension_subset = self.eigen_variations[:, :n_vectors].T
            subset_covariances = [self.var2cov(x) for x in dimension_subset]

            max_differences.append(
                np.abs(np.real(self.cov - np.sum(subset_covariances, axis=0))).max()
            )

        return max_differences

    @staticmethod
    def var2cov(mat) -> np.ndarray:
        return np.outer(mat, mat)

    def find_important_eigendimension_indices(self):

        important_dims = (
            np.asarray(self.max_differences) / self.cov.max() > self.precision
        )

        self.N_important_dims = np.sum(important_dims)
        self.important_dims_indices = important_dims
        logging.info(
            f"Found that %s eigendirections matter for %s per cent precision",
            self.N_important_dims,
            self.precision,
        )

    def _get_unrolled_variations(self):

        return self.eigen_variations[:, self.important_dims_indices].reshape(
            len(self.decay_modes),
            len(self.fit_ctgies),
            self.Nbins,
            self.N_important_dims,
        )

    def _get_unrolled_nominals(self):
        return self.nominal_hist.reshape(
            len(self.decay_modes), len(self.fit_ctgies), self.Nbins
        )
