from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sysvar.corrections import BaseCorrection
from sysvar.visualize import VariatorVisualizer


class MissingSavingInfo(Exception):
    pass


# TODO should this really be an ABC ? Need to think about it...
class Variator(ABC):
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

        self.correction = correction
        self.Nvar = Nvar
        self.variations = self.get_correction_variations()

        self._save_figures = False
        self.visualizer = None
        self.figure_save_info = {
            "namespace": None,
            "top_dir": None,
            "dir_spec": None,
            "extra_ext": None,
            "save": None,
        }

    @property
    def save_figures(self):
        return self._save_figures

    @save_figures.setter
    def save_figures(self, value):

        if not isinstance(value, bool):
            raise TypeError(
                "save_figures is strictly a boolean variable. Please pass True or False. save_figure defaults to False"
            )
        self._save_figures = value

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

    def register_figure_saving_info(
        self,
        namespace: list = None,
        top_dir: str = None,
        dir_spec: str | None | bool = None,
        extra_ext: str | Iterable | None = None,
    ):

        self.figure_save_info = {
            "namespace": namespace,
            "top_dir": top_dir,
            "dir_spec": dir_spec,
            "extra_ext": extra_ext,
            "save": self._save_figures,
        }

    @staticmethod
    def explain_figure_saving_info():
        raise NotImplementedError(
            "Implement the method that explains what namespace, top_dir, dir_spec and extra_ext are doing"
        )

    def _check_saving_status(self):
        if self._save_figures:
            if all(x is None for x in list(self.figure_save_info.values())):
                raise MissingSavingInfo(
                    "You wish to save your figures by setting save_figures = True, but you have not specified the target saving info. SysVar will not save the figures at a random directory. Please call the register_figure_saving_info method to specify the necessary information before replotting. If you're are unsure what this info should be, call the explain_figure_saving_info method for a quick overview"
                )
            else:
                pass

    def plot_cov_and_corr(self):
        self._check_saving_status()
        visualizer = VariatorVisualizer(self, **self.figure_save_info)
        visualizer.plot_cov_and_corr()

    def plot_gaussian_variations(self):
        self._check_saving_status()
        visualizer = VariatorVisualizer(self, **self.figure_save_info)
        visualizer.plot_gaussian_variations()

    def plot_relative_variations_in_grid(self, nbins: int = 41):

        self._check_saving_status()
        visualizer = VariatorVisualizer(self, **self.figure_save_info)
        visualizer.plot_relative_variations_in_grid(nbins)
