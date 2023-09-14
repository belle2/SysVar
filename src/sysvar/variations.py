from abc import ABC, abstractmethod
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


class UncertaintyWithSameNameExists(Exception):
    pass


# TODO should this really be an ABC ? Need to think about it...
class Variator(ABC):

    """
    Abstract base class for generating variations on a correction.
    Uncertainty objects need to be appended to the variator.
    Then the variator will create a big covariance matrix of all the uncertainties.
    Through the get_variations_from_uncertainty method, one can examine the effect
    of the variations coming from a specific uncertainty.

    Args:
        correction (Correction): The correction object.

    Attributes:
        correction (Correction): The correction object.
        uncertainties (dict): A dictionary to store uncertainties associated with the correction.

    """

    def __init__(self, correction: Correction):

        """
        Initialize a Variator with a correction object.

        Args:
            correction (Correction): The correction object.

        """

        self.correction = correction
        self.uncertainties = {}

    def add_uncertainty(self, unc: Uncertainty) -> None:

        """
        Add an uncertainty to the Variator.

        Args:
            unc (Uncertainty): The uncertainty to be added.

        Raises:
            UncertaintyWithSameNameExists: If uncertainty with the same name has already been added to the variator.

        """
        if unc.name in self.uncertainties.keys():
            raise UncertaintyWithSameNameExists(
                f"An uncertainty with the name {unc.name} already exist in the set of uncertainties that the variator will consider. Make sure that you add a specific uncertainty only once, and that there are no duplicate names"
            )
        else:
            self.uncertainties.update({unc.name: unc})

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
        zeros = np.zeros(len(self.correction.values))

        # Generate the up or down variations based on a standard normal
        return np.random.multivariate_normal(zeros, covariance, Nvar)

    def get_correction_variations(self, Nvar: int) -> np.ndarray:

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
        variations = self.generate_variations(Nvar, total_cov_matrix)

        return self.correction.values + variations

    def get_variations_from_uncertainty(self, Nvar: int, name: str) -> np.ndarray:
        """
        Helper function to inspect variations coming from a single source of uncertainty.

        Args:
            Nvar (int): The number of variations to generate.
            name (str): The name of the uncertainty.

        Returns:
            np.ndarray: An array of variations from the specified uncertainty.

        """

        return self.generate_variations(Nvar, self.uncertainties[name].cov_matrix)

    def visualize_variations(self, Nvar: int = 5):

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

        x_edges = np.unique(
            np.concatenate((self.correction.lower_bounds, self.correction.upper_bounds))
        )

        # Plot the nominal weights
        plot_variation_on_axis(ax, x_edges, self.correction.values)

        variations = self.get_correction_variations(Nvar)

        for i in range(Nvar):
            plot_variation_on_axis(
                ax=ax, x=x_edges, variation=variations[i, :], index=i
            )

        return fig, ax
