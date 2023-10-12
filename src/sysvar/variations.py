from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from sysvar.corrections import Correction
from sysvar.visualize import (
    plot_matrix_on_axis,
    plot_variation_on_axis,
    plot_gaussian_variation_on_axis,
    create_single_figure,
    create_double_figure,
    create_triple_figure,
)

# TODO should this really be an ABC ? Need to think about it...
class Variator(ABC):

    """
    Abstract base class for generating variations on a correction.
    Then the variator will create a big covariance matrix of all the uncertainties.
    Through the get_variations_from_uncertainty method, one can examine the effect
    of the variations coming from a specific uncertainty.

    Args:
        correction (Correction): The correction object.
        Nvar (int): The number of variations to be generated. Defaults to 20.

    Attributes:
        correction (Correction): The correction object.
        uncertainties (dict): A dictionary to store uncertainties associated with the correction.

    """

    def __init__(self, correction: Correction, Nvar: int = 20):

        """
        Initialize a Variator with a correction object.

        Args:
            correction (Correction): The correction object.

        """

        self.correction = correction
        self.uncertainties = self.correction.uncertainties
        self.Nvar = Nvar

    @property
    def variations(self) -> np.ndarray:
        return self.get_correction_variations()

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._build_total_covariance()

    @property
    def corr_matrix(self) -> np.ndarray:
        std_devs = np.sqrt(np.diag(self.cov_matrix))
        return self.cov_matrix / np.outer(std_devs, std_devs)

    def _build_total_covariance(self) -> np.ndarray:

        """
        Build the total covariance matrix from all added uncertainties.
        Here all the covariance matrices from all uncertainties are summed up
        to a total covariance matrix

        Returns:
            np.ndarray: The total covariance matrix.

        """

        return np.add(*[unc.cov_matrix for unc in self.uncertainties.values()])

    def generate_variations(self, Nvar: int, covariance: np.ndarray) -> np.ndarray:
        """
        Generate variations based on a covariance matrix.
        This is using a standard multivariate normal.

        Args:
            Nvar (int): The number of variations to generate.
            covariance (np.ndarray): The covariance matrix.

        Returns:
            np.ndarray: An array of variations.

        """
        # Create a zero-ed matrix to get the dimensions right
        zeros = np.zeros(len(self.correction.central_values))

        # Generate the up or down variations based on a standard normal
        return np.random.multivariate_normal(zeros, covariance, Nvar)

    def get_correction_variations(self) -> np.ndarray:

        """
        Get variations on the correction.
        This adds the generated variations from the total covariance matrix
        to the nominal values of the correction.

        Args:
            Nvar (int): The number of variations to generate.

        Returns:
            np.ndarray: An array of correction variations.

        """

        total_cov_matrix = self._build_total_covariance()
        variations = self.generate_variations(self.Nvar, total_cov_matrix)

        return self.correction.central_values + variations

    def get_variations_from_uncertainty(self, Nvar: int, name: str) -> np.ndarray:
        """
        Helper function to inspect variations coming from a single source of uncertainty.
        Creates new variations but likely this is okay as we're interested in examining
        the sources qualitatevely.

        Args:
            Nvar (int): The number of variations to generate.
            name (str): The name of the uncertainty.

        Returns:
            np.ndarray: An array of variations from the specified uncertainty.

        """

        return self.generate_variations(Nvar, self.uncertainties[name].cov_matrix)

    def visualize_relative_variations(self, Nvar: int = 5):

        """
        Visualize variations of the correction.
        Plots the relative variations of the templates.
        The Nvar argument specifies the number of variatios that will be plotted.
        Defaults to 5.

        Args:
            Nvar (int, optional): The number of variations to visualize.

        Returns:
            Tuple[Figure, Axis]: A tuple containing the figure and axis objects.

        """

        fig, ax = create_single_figure()

        variations = self.get_correction_variations()

        for i in range(Nvar):
            plot_variation_on_axis(
                ax=ax,
                x=self.correction.value_edges,
                variation=variations[i, :] / self.correction.central_values,
                index=i,
                plot_func="stairs",
            )

        return fig, ax

    def visualize_gaussian_variations(self):

        if len(self.correction.central_values) == 3:
            fig, ax = create_triple_figure()
        else:
            raise NotImplementedError(
                "Need to figure out how to deal with multiple number of figures"
            )

        for i, c in enumerate(self.correction.central_values):
            plot_gaussian_variation_on_axis(
                ax[i], c, self.variations[:, i], self.correction.strings[i]
            )
        return fig, ax

    def visualize_covariance(self):

        fig, ax = create_single_figure()

        plot_matrix_on_axis(
            ax,
            self.cov_matrix,
            self.correction.string_boundaries,
            "Covariance matrix",
            "Correction bins",
        )

        return fig, ax

    def visualize_covariance_and_correlation(self):

        fig, ax = create_double_figure()

        plot_matrix_on_axis(
            ax[0],
            self.cov_matrix,
            self.correction.strings,
            "Covariance matrix",
            "Correction bins",
        )

        plot_matrix_on_axis(
            ax[1],
            self.corr_matrix,
            self.correction.strings,
            "Correlation matrix",
            "Correction bins",
        )

        return fig, ax
