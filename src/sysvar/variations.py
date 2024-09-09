from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

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

    def __init__(self, correction: BaseCorrection, Nvar: int = 20):
        """
        Initialize a Variator with a correction object.

        Args:
           central_values (Iterable): The central values of the correction weights

        """
        super().__init__()
        self.correction = correction
        self.Nvar = Nvar
        self.variations = self.get_correction_variations()

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
        if len(self.correction.uncertainties.values()) > 1:
            return np.add(
                *[unc.cov_matrix for unc in self.correction.uncertainties.values()]
            )
        else:
            return next(iter(self.correction.uncertainties.values())).cov_matrix

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

        variations = self.generate_variations(self.Nvar, self.cov_matrix)

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

        return self.generate_variations(
            Nvar, self.correction.uncertainties[name].cov_matrix
        )

    def plot_cov_and_corr(self, save: bool = False, filename: str = ""):
        visualizer = VariatorVisualizer(self)
        visualizer.plot_cov_and_corr(save=save, filename=filename)

    def plot_gaussian_variations(self, save: bool = False, filename: str = ""):
        self._check_saving_status()
        visualizer = VariatorVisualizer(self)
        visualizer.plot_gaussian_variations(save=save, filename=filename)

    def plot_relative_variations_in_grid(
        self, nbins: int = 41, save: bool = False, filename: str = ""
    ):

        self._check_saving_status()
        visualizer = VariatorVisualizer(self)
        visualizer.plot_relative_variations_in_grid(nbins, save=save, filename=filename)
