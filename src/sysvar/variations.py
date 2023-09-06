from abs import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from sysvar.uncertainties import Uncertainty
from sysvar.corrections import Correction
from sysvar.visualize import (
    plot_matrix_on_axis,
    plot_variation_on_axis,
    create_double_figure,
    create_single_figure,
)


class Variator(ABC):
    def __init__(self, correction: Correction):

        self.correction = correction
        self.uncertainties = {}
        self.total_cov_matrix = self._build_total_covariance()

    def add_uncertainty(self, unc: Uncertainty) -> None:

        self.uncertainties.update({unc.name: unc})

    def _build_total_covariance(self) -> np.ndarray:

        return np.add(*[unc.cov_matrix for unc in self.uncertainties.values()])

    def generate_variations(self, Nvar: int, covariance: np.ndarray) -> np.ndarray:
        # Create a zero-ed matrix to get the dimensions right
        zeros = np.zeros(len(self.correction.values))

        # Generate the up or down variations based on a standard normal
        return np.random.multivariate_normal(zeros, covariance, Nvar)

    def get_correction_variations(self, Nvar: int) -> np.ndarray:

        variations = self.generate_variations(Nvar, self.total_cov_matrix)

        return self.correction.values + variations

    def get_variations_from_uncertainty(self, Nvar: int, name: str) -> np.ndarray:
        """Helper function to inspect variations coming from a single source of uncertainty"""

        return self.generate_variations(Nvar, self.uncertainties[name].cov_matrix)

    def visualize_variations(self, Nvar: int = 5):

        fig, ax = create_single_figure()

        x_edges = np.unique(
            np.concatenate((self.correction.lower_bounds, self.correction.upper_bounds))
        )

        # Plot the nominal weights
        plot_variation_on_axis(
            ax, x_edges, self.correction.values, "Nominal weight", "black", "dashed"
        )

        variations = self.get_correction_variations(Nvar)
        # TODO move this from here...
        import seaborn as sns

        palette = sns.color_palette("colorblind")

        for i in range(Nvar):
            plot_variation_on_axis(
                ax, x_edges, variations[i, :], f"variation {i}", palette[i], "solid"
            )

        return fig, ax
