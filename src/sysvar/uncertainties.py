from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Union

import numpy as np

from visualize import plot_matrix_on_axis, create_double_figure, create_single_figure
from corrections import Correction


class NotAnArrayError(Exception):
    pass


class MultiDimArrayError(Exception):
    pass


class NonMatchingCorrections(Exception):
    pass


class Uncertainty(ABC):
    """
    Abstract base class for representing uncertainties with a name and error values.

    Args:
        name (str): The name of the uncertainty.
        errors (np.ndarray): An array containing the error values.

    Attributes:
        name (str): The name of the uncertainty.
        errors (np.ndarray): An array containing the error values.
        cov_matrix (np.ndarray): The covariance matrix of the uncertainty.

    """

    def __init__(self, correction: Correction, name: str, errors: np.ndarray):
        """
        Initialize an Uncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (np.ndarray): An array containing the error values.

        Raises:
            NotAnArrayError: If errors is not an instance of np.ndarray.
            MultiDimArrayError: If errors is not a 1D array.
            NonMatchingCorrections: If the length of errors and corrections
               are unequal.

        """
        self.correction = correction
        self.name = name
        if self._is_valid_input_errors(errors):
            self.errors = errors
        self.cov_matrix = self.build_covariance()
        super().__init__()

    @abstractmethod
    def build_covariance(self) -> np.ndarray:
        """
        Abstract method to build the covariance matrix of the uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        pass

    def _is_valid_input_errors(self, errors) -> Union[None, bool]:
        """
        Validate the input errors.

        Args:
            errors: The input error values.

        Returns:
            Union[None, bool]: True if the errors are valid, otherwise None.

        Raises:
            NotAnArrayError: If errors is not an instance of np.ndarray.
            MultiDimArrayError: If errors is not a 1D array.
            NonMatchingCorrections: If the length of errors and corrections
               are unequal.

        """
        if not isinstance(errors, np.ndarray):
            raise NotAnArrayError("The errors must be provided as np arrays")

        elif errors.ndim > 1:
            raise MultiDimArrayError("The errors must be provided as a 1D array")

        elif len(errors) != len(self.correction.values):
            raise NonMatchingCorrections(
                f"The corrections have length of {len(self.corrections.values)}, but you are trying to pass uncertainties of length {len(errors)}."
            )
        else:
            return True

    def visualize_covariance(self):

        fig, ax = create_single_figure()

        plot_matrix_on_axis(
            ax,
            self.cov_matrix,
            self.correction.build_strings(),
            f"{self.name} Covariance matrix",
        )

        return fig, ax

    def visualize_covariance_and_correlation(self):

        fig, ax = create_double_figure()

        plot_matrix_on_axis(
            ax[0],
            self.cov_matrix,
            self.correction.build_strings(),
            f"{self.name} Covariance matrix",
        )

        plot_matrix_on_axis(
            ax[1],
            self.corr_matrix,
            self.correction.build_strings(),
            f"{self.name} Correlation matrix",
        )

        return fig, ax


class FullyCorrelatedUncertainty(Uncertainty):
    """
    Represents a fully correlated uncertainty with a name and error values.

    This class inherits from the Uncertainty base class and implements a fully correlated
    uncertainty with a correlation matrix of ones.

    Args:
        name (str): The name of the uncertainty.
        errors (Iterable): An iterable containing the error values.

    Attributes:
        name (str): The name of the uncertainty.
        errors (np.ndarray): An array containing the error values.
        cov_matrix (np.ndarray): The covariance matrix of the uncertainty.
        corr_matrix (np.ndarray): The correlation matrix of the uncertainty.

    """

    def __init__(self, correction: Correction, name: str, errors: Iterable):
        """
        Initialize a FullyCorrelatedUncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (Iterable): An iterable containing the error values.

        """
        self.corr_matrix = np.ones((len(errors), len(errors)))
        super().__init__(correction, name, np.array(errors))

    def build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix of the fully correlated uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        return np.diag(self.errors) @ self.corr_matrix @ np.diag(self.errors)


class UncorrelatedUncertainty(Uncertainty):
    """
    Represents an uncorrelated uncertainty with a name and error values.

    This class inherits from the Uncertainty base class and implements an uncorrelated
    uncertainty with a correlation matrix of identity.

    Args:
        name (str): The name of the uncertainty.
        errors (Iterable): An iterable containing the error values.

    Attributes:
        name (str): The name of the uncertainty.
        errors (np.ndarray): An array containing the error values.
        cov_matrix (np.ndarray): The covariance matrix of the uncertainty.
        corr_matrix (np.ndarray): The correlation matrix of the uncertainty.

    """

    def __init__(self, correction: Correction, name: str, errors: Iterable):
        """
        Initialize an UncorrelatedUncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (Iterable): An iterable containing the error values.

        """
        self.corr_matrix = np.identity(len(errors))
        super().__init__(correction, name, np.array(errors))

    def build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix of the uncorrelated uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        return np.diag(self.errors * self.errors)


class CorrelatedUncertainty(Uncertainty):
    def __init__(self):

        raise NotImplementedError(
            "Only fully correlated or uncorrelated uncertainties so far"
        )
