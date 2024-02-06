from functools import cached_property

import numpy as np
from pandas import DataFrame

import uproot

from sysvar.corrections import Correction
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

        self.correction = Correction(
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

    def _get_varied_templates(self):

        varied_templates = []
        for reco_mode in self.decay_modes:
            logging.info("Building in reco region %s", str(reco_mode[1]))
            reco_df = self.df.query("Bsig_dmID in @reco_mode[0]")
            for fit_ctgy in self.fit_ctgies:
                logging.info("Building template %s", str(fit_ctgy))

                tmp_df = reco_df.query(f"fit_ctgy == {fit_ctgy}")
                # Skip template create if the query yields an empty dataframe
                if len(tmp_df) < 1:
                    continue

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

    @staticmethod
    def var2cov(mat) -> np.ndarray:
        return np.outer(mat, mat)

    def find_important_eigendimension_indices(self, precision: float = 0.01):

        total_N_vectors = len(self.eigen_vectors)
        max_differences = []
        for n_vectors in range(total_N_vectors):

            dimension_subset = self.eigen_variations[:, : n_vectors + 1].T
            subset_covariances = [self.var2cov(x) for x in dimension_subset]

            max_differences.append(
                np.abs(np.real(self.cov - np.sum(subset_covariances, axis=0))).max()
            )

        important_dims = np.asarray(max_differences) > precision

        self.N_important_dims = np.sum(important_dims)
        self.important_dims_indices = important_dims

        return important_dims

    def _get_unrolled_variations(self):

        # PATCH, FIXME, TODO, align fit ctgies and read them
        return self.eigen_variations[:, self.important_dims_indices].reshape(
            len(self.decay_modes), 6, self.Nbins, self.N_important_dims
        )

    def _get_unrolled_nominals(self):
        # PATCH, FIXME, TODO, align fit ctgies and read them
        return self.nominal_hist.reshape(len(self.decay_modes), 6, self.Nbins)

    def save_variations(self):

        variations = self._get_unrolled_variations()
        nominals = self._get_unrolled_nominals()

        with uproot.update(self.settings["filename"]) as newfile:
            logging.info("Updating file with uproot: %s", self.settings["filename"])

            # PATCH, FIXME, TODO, align fit ctgies and read them
            for i_rm, reco_mode in enumerate(self.decay_modes):
                if reco_mode[0] in [[51103, 51104], [52103, 52104]]:
                    template_ids = [3, 4, 5, 6, 7, 8]
                elif reco_mode[0] in [[51101], [51102], [52101], [52102]]:
                    template_ids = [1, 2, 3, 4, 7, 8]

                for j_ctgy, ctgy in enumerate(template_ids):
                    for k_var in range(self.N_important_dims):
                        logging.info(
                            "Computing template in region: %s for fit ctgy template: %s and variation #%s",
                            reco_mode[1],
                            ctgy,
                            k_var + 1,
                        )
                        newfile[
                            f"{reco_mode[1]}/{ctgy}/{self.syst_effect}_var{k_var+1}"
                        ] = (
                            nominals[i_rm, j_ctgy, :]
                            + variations[i_rm, j_ctgy, :, k_var],
                            np.linspace(0, 1, self.Nbins + 1),
                        )
