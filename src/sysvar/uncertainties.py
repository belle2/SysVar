from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from sysvar.utils import SavableAttributesObject

import numpy as np


def get_uncertainty_types():

    return {
        "fully_correlated": FullyCorrelatedUncertainty,
        "uncorrelated": UncorrelatedUncertainty,
        "explicitly_correlated": ExplicitlyCorrelatedUncertainty,
        # "fully_correlated_in_parts": FullyCorrelatedUncertaintyInParts,
    }


class NotAnArrayError(Exception):
    pass


class MultiDimArrayError(Exception):
    pass


class EmptyArrayError(Exception):
    pass


class ValueError(Exception):
    pass


class Uncertainty(ABC, SavableAttributesObject):
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

    def __init__(self, name: str, errors: np.ndarray, string_boundaries: List):
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
        self.name = name
        if self._is_valid_input_errors(errors):
            self.errors = errors
        self.string_boundaries = string_boundaries
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

    def _is_valid_input_errors(self, errors) -> None | bool:
        """
        Validate the input errors.

        Args:
            errors: The input error values.

        Returns:
            None | bool: True if the errors are valid, otherwise None.

        Raises:
            NotAnArrayError: If errors is not an instance of np.ndarray.
            MultiDimArrayError: If errors is not a 1D array.
            EmptyArrayError: If errors is an empty array.
            NonMatchingCorrections: If the length of errors and corrections
               are unequal.

        """
        if not isinstance(errors, (np.ndarray, list)):
            raise NotAnArrayError("The errors must be provided as np arrays")

        elif np.array(errors).ndim > 1:
            raise MultiDimArrayError("The errors must be provided as a 1D array")

        elif len(errors) == 0:
            raise EmptyArrayError("The errors array cannot be empty")

        if not np.isfinite(errors).all():
            raise ValueError("The errors array contains invalid values")
        else:
            return True


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

    def __init__(self, name: str, errors: Iterable, string_boundaries: List):
        """
        Initialize a FullyCorrelatedUncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (Iterable): An iterable containing the error values.

        """
        self._is_valid_input_errors(errors)
        self.corr_matrix = np.ones((len(errors), len(errors)))
        super().__init__(name, np.array(errors), string_boundaries)

    def build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix of the fully correlated uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        return np.diag(self.errors) @ self.corr_matrix @ np.diag(self.errors)


class FullyCorrelatedUncertaintyInParts(Uncertainty):
    """
    Represents a fully correlated uncertainty only in parts with a name and error values.

    This class inherits from the Uncertainty base class and implements a fully correlated
    uncertainty with a correlation matrix that is full of ones in some regions and
    completely uncorrelated in other regions

    Args:
        name (str): The name of the uncertainty.
        errors (Iterable): An iterable containing the error values.
        part = dimensions (Iterable): dimensions of correlated/uncorrelated parts

    Attributes:
        name (str): The name of the uncertainty.
        errors (np.ndarray): An array containing the error values.
        cov_matrix (np.ndarray): The covariance matrix of the uncertainty.
        corr_matrix (np.ndarray): The correlation matrix of the uncertainty.

    """

    def __init__(
        self, name: str, errors: Iterable, string_boundaries: List, part_dimensions
    ):
        """
        Initialize a FullyCorrelatedUncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (Iterable): An iterable containing the error values.

        """
        raise NotImplementedError(
            "FullyCorrelatedUncertaintyInParts is not supported currently. This would be useful when we have corrections which are fully correlated in some regions and completely uncorrelated in other regions. e.g. Corrections for Run1 MC and Run2 MC separately."
        )
        self._is_valid_input_errors(errors)
        self.part_dimensions = part_dimensions
        self.corr_matrix = self.build_correlation_matrix()
        super().__init__(name, np.array(errors), string_boundaries)

    def build_correlation_matrix(self):

        # Initialize the matrix with zeros
        matrix = np.zeros((sum(self.part_dimensions), sum(self.part_dimensions)))

        # Track the starting row and column indices for each part
        row_start = 0
        col_start = 0

        # Iterate over each part
        for part_dim in self.part_dimensions:
            # Assign values to the current part
            matrix[
                row_start : row_start + part_dim, col_start : col_start + part_dim
            ] = 1

            # Update starting row and column indices for the next part
            row_start += part_dim
            col_start += part_dim

        return matrix

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

    def __init__(self, name: str, errors: Iterable, string_boundaries: List):
        """
        Initialize an UncorrelatedUncertainty instance with a name and error values.

        Args:
            name (str): The name of the uncertainty.
            errors (Iterable): An iterable containing the error values.

        """
        self._is_valid_input_errors(errors)
        self.corr_matrix = np.identity(len(errors))
        super().__init__(name, np.array(errors), string_boundaries)

    def build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix of the uncorrelated uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        return np.diag(self.errors * self.errors)


class ExplicitlyCorrelatedUncertainty(Uncertainty):
    def __init__(
        self,
        name: str,
        errors: np.ndarray,
        string_boundaries: List[str],
        explicit_cov_matrix: np.ndarray | None = None,
    ):

        self._is_valid_input_errors(errors)
        self.explicit_cov_matrix = explicit_cov_matrix
        super().__init__(name, errors, string_boundaries)
        self.corr_matrix = self.build_correlation_matrix()
        # self._cov_matrix = (
        #     cov_matrix
        #     if cov_matrix is not None
        #     else np.diag(np.square(self.errors))  # Fallback to uncorrelated
        # )

    def build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix of the explicitly correlated uncertainty.

        Returns:
            np.ndarray: The covariance matrix of the uncertainty.

        """
        return self.explicit_cov_matrix

    def build_correlation_matrix(self):
        """
        Build the correlation matrix of the explicitly correlated uncertainty.

        Returns:
            np.ndarray: The correlation matrix of the uncertainty.

        """
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix is not defined.")

        return self.cov_matrix / np.outer(self.errors, self.errors)
