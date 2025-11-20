from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable
import matplotlib.pyplot as plt

import numpy as np

from sysvar.corrections import BaseCorrection
from sysvar.visualize import VariatorVisualizer
from sysvar.utils import SavableAttributesObject


class MissingSavingInfo(Exception):
    pass


# TODO should this really be an ABC ? Need to think about it...
class Variator(ABC, SavableAttributesObject):
    """
    Abstract base class for generating variations on a correction.
    Then the variator will create a big covariance matrix of all the uncertainties.
    Through the get_variations_from_uncertainty method, one can examine the effect
    of the variations coming from a specific uncertainty.

    Args:
        correction (BaseCorrection): The correction object.
        Nvar (int): The number of variations to be generated. Defaults to 20.

    Attributes:
        correction (BaseCorrection): The correction object.
        uncertainties (dict): A dictionary to store uncertainties associated with the correction.

    """

    def __init__(self, correction: BaseCorrection, Nvar: int = 20, seed: int = 8311311):
        """
        Initialize a Variator with a correction object.

        Args:
           central_values (Iterable): The central values of the correction weights

        """
        super().__init__()
        self.correction = correction
        self.Nvar = Nvar
        self._variations = self.get_correction_variations(seed=seed)

    @property
    def variations(self) -> np.ndarray:
        return self._variations

    @variations.setter
    def variations(self, variations: np.ndarray):
        """
        Setter for variations. This is used to set the variations manually.
        This is useful if the user wants to set the variations manually
        instead of generating them from the covariance matrix.

        Args:
            variations (np.ndarray): The variations to be set.

        """
        if not isinstance(variations, np.ndarray):
            raise TypeError("Variations must be a numpy array.")
        self._variations = variations

    @property
    def cov_matrix(self) -> np.ndarray:
        """
        Build the total covariance matrix from all added uncertainties.
        Here all the covariance matrices from all uncertainties are summed up
        to a total covariance matrix
        Return the covariance matrix from one uncertainty type of only one is present

        Returns:
            np.ndarray: The total covariance matrix.

        """
        # If we have already set the covariance matrix through the setter return this.
        # This implies that the user knows what they're doing
        if hasattr(self, "_cov_matrix") and self._cov_matrix is not None:
            return self._cov_matrix
        # If the setter has not been used
        # either read the covariance matrix from the correction
        elif (
            hasattr(self.correction, "cov_matrix")
            and self.correction.cov_matrix is not None
        ):
            return self.correction.cov_matrix
        # or build the covariance matrix from the uncertainties
        else:
            if len(self.correction.uncertainties.values()) > 1:
                return np.sum(
                    [unc.cov_matrix for unc in self.correction.uncertainties.values()],
                    axis=0,
                )
            else:
                return next(iter(self.correction.uncertainties.values())).cov_matrix

    # Useful for covariance matrix calculations for custom histgrams e.g. Data/MC plots
    @cov_matrix.setter
    def cov_matrix(self, cov_matrix):
        self._cov_matrix = cov_matrix

    @property
    def corr_matrix(self) -> np.ndarray:
        std_devs = np.sqrt(np.diag(self.cov_matrix))
        return self.cov_matrix / np.outer(std_devs, std_devs)

    @property
    def relative_variations(self) -> np.ndarray:

        return np.divide(
            self.variations,
            np.array(self.correction.central_values),
            out=np.zeros_like(self.variations),
            where=np.array(self.correction.central_values) != 0,
        )

    def generate_variations(
        self, Nvar: int, covariance: np.ndarray, seed: int = 8311311
    ) -> np.ndarray:
        """
        Generate variations based on a covariance matrix.
        This is using a standard multivariate normal.

        Args:
            Nvar (int): The number of variations to generate.
            covariance (np.ndarray): The covariance matrix.

        Returns:
            np.ndarray: An array of variations.

        """
        rng = np.random.default_rng(seed)

        # Create a zero-ed matrix to get the dimensions right
        zeros = np.zeros(len(self.correction.central_values))

        # Generate the up or down variations based on a standard normal
        return rng.multivariate_normal(zeros, covariance, Nvar)

    def get_correction_variations(self, seed: int = 8311311) -> np.ndarray:
        """
        Get variations on the correction.
        This adds the generated variations from the total covariance matrix
        to the nominal values of the correction.

        Args:
            Nvar (int): The number of variations to generate.

        Returns:
            np.ndarray: An array of correction variations.

        """

        variations = self.generate_variations(self.Nvar, self.cov_matrix, seed=seed)

        return self.correction.central_values + variations

    def get_variations_from_uncertainty(
        self, Nvar: int, name: str, seed: int = 8311311
    ) -> np.ndarray:
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

        return self.generate_variations(
            Nvar, self.correction.uncertainties[name].cov_matrix, seed=seed
        )

    def plot_cov_and_corr(
        self, save: bool = False, filename: str = ""
    ) -> tuple[plt.Figure, plt.Axes]:
        visualizer = VariatorVisualizer(self)
        fig, ax = visualizer.plot_cov_and_corr(save=save, filename=filename)
        return fig, ax

    def plot_gaussian_variations(self, save: bool = False, filename: str = ""):
        visualizer = VariatorVisualizer(self)
        visualizer.plot_gaussian_variations(save=save, filename=filename)

    def plot_relative_variations_in_grid(
        self, nbins: int = 41, save: bool = False, filename: str = ""
    ) -> tuple[plt.Figure, plt.Axes]:

        visualizer = VariatorVisualizer(self)
        fig, ax = visualizer.plot_relative_variations_in_grid(
            nbins=nbins, save=save, filename=filename
        )
        return fig, ax
