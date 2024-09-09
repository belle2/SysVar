from __future__ import annotations

from os import path
from pathlib import Path
from yaml import safe_load
from typing import List
import numpy as np

from abc import ABC, abstractmethod


import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)

# This return [/*****/Sysvar/src/sysvar]
from sysvar import __path__


def read_yaml(cfg_name: str, deeper_dir: str = ""):
    """Reads a yaml file file from the configuration directory of the repository and
    and returns a dictionary with the yaml data

    Args:
    cfg_name: Name of the yaml file which is located in the configs directory
    deeper_dir: Specify the directory within the configs directory.

    Returns:
    Dictionary read from the yaml file
    """
    # Get the parent directory of the repository
    parent_dir = _get_parent_dir()
    # Get the correct configs directory
    configs_dir = _get_configs_dir(deeper_dir)
    # Get config file name
    file_name = _get_config_file_name(cfg_name)

    # Join the parent dir of the framework, the configs dir and the file name
    with open(path.join(parent_dir, configs_dir, file_name), "r") as file:
        # Read the yaml file into a dict
        cfg = safe_load(file)

    return cfg


def _get_src_dir():
    """Extracts the src path of the package from the __path__ list."""
    return Path(__path__[0])


def _get_parent_dir():
    """Extracts the parent path of the package from the __path__ list."""
    # Returns the grandparent path of __path__
    return Path(__path__[0]).parents[1]


def _get_configs_dir(deeper_dir: str) -> str:
    """Helper function to get the correct config file directory
    This helps for further categoriazation of config files

    Args:
    deeper_dir: Specify the directory within the configs directory.

    Returns:
    Specified directory within the configs directory of the framework
    """
    return path.join("configs", deeper_dir)


def _get_config_file_name(cfg_name: str) -> str:
    """Builds and returns a config file name

    Args:
    cfg_name: name of the config file

    Returns:
    The name of the config file with the added .yaml extension
    """
    return ".".join((cfg_name, "yaml"))


def corr2cov(corr, var):
    """Calculates the covariance matrix from a given
    correlation matrix and a variance vector.

    Arguments
    ---------
    corr : np.ndarray
        Correlation matrix of shape (n,n).
    var : np.ndarray
        Variance vector of shape (n,).

    Return
    ------
    out : np.ndarray
        Covariance matrix. Shape is (n,n).
    """
    D = np.diag(var)
    return np.matmul(D, np.matmul(corr, D))


def cov2corr(covariance):
    """
    Compute the correlation matrix from the given covariance matrix.

    Parameters:
    covariance : numpy.ndarray
        The covariance matrix.

    Returns:
    numpy.ndarray
        The correlation matrix.
    """
    v = np.sqrt(np.diag(covariance))  # Compute the standard deviation for each variable
    outer_v = np.outer(v, v)  # Compute the outer product of the standard deviations
    correlation = (
        covariance / outer_v
    )  # Divide the covariance matrix by the outer product of standard deviations
    correlation[covariance == 0] = 0  # Set correlations with zero covariance to zero
    return np.real(correlation)  # Return the real part of the correlation matrix


def get_varied_FF_central_values(model):
    """
    Inhereted from Henrik
    Function to calculate the different +-1sigma variations of the form factors.
    For that, the form factor parameters are rotated into a paramter space in which all of
    them are perfectly orthogonal and uncorrelated (the eigenvectors of the covariance matrix).
    Then, the 1 sigma up and down variation of each of these new parameters is taken (symbolized
    by the eigenvalues) and this is rotated back into the original parameter space.
    One gets 2*N different sets of varied central values (for N parameters of the FF model), each
    presenting the 1sigma up- and down variation of one of the orthogonal parameters.
    Subsequently, these new varied sets can be used to derive the 1 sigma varied histogram with
    eFFORT or HAMMER.
    :param ff_dict: Dictionary that includes the form factor model parameters and their correlations.
                    The parameters need to be of listed with param_{i} and uncert_{i}.
                    Moreover the dict needs to contain the "corrmat" entry which includes the
                    correlation matrix with the correct paramter order and the "num_param" entry
                    which specifies the number of parameters in this form factor model.
                    If statistical and systematic errors are distinguished, the dict needs entries
                    "stat_{i}", "syst_{i}" and corrmat_stat / _syst.
    """
    # get central values covariance matrices:
    cvs = np.array(
        [
            param.nominal_value if name != "DelMbc" else param["DelMbc"].nominal_value
            for name, param in model.params.items()
        ]
    )
    errors = np.array(
        [
            param.std_dev if name != "DelMbc" else param["DelMbc"].std_dev
            for name, param in model.params.items()
        ]
    )
    covariance_matrix = corr2cov(model.corr_matrix, errors)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    diagonalized_covariance_matrix = (
        np.linalg.inv(eigenvectors) @ covariance_matrix @ eigenvectors
    )

    # check if one of the parameters is DelMbc,
    # in this case, unfortunately the charm mass mc needs
    # to be calculated and must replace DelMbc to still fulfil the parameter string
    # of the dictionary.
    param_names = [name for name in model.params.keys()]
    if "DelMbc" in param_names:
        logging.warn(
            "WARNING! The selected form factor model contains mb and mc. Please take extra care that everything works as expected."
        )
        mb_index = param_names.index("mb")
        DelMbc_index = param_names.index("DelMbc")

    eigenvalue_variations = []
    for i, eigenvalue in enumerate(eigenvalues):
        variation = np.zeros(len(cvs))
        variation[i] = np.sqrt(eigenvalue)

        up = cvs + eigenvectors @ variation
        down = cvs - eigenvectors @ variation

        if "DelMbc" in param_names:
            logging.info(
                f"Calculate mc for up (mc = {up[mb_index]:.3f} - {up[DelMbc_index]:.3f}) and down ({down[mb_index]:.3f} - {down[DelMbc_index]:.3f}) variations."
            )
            up[DelMbc_index] = up[mb_index] - up[DelMbc_index]
            down[DelMbc_index] = down[mb_index] - down[DelMbc_index]
            logging.info(
                f"Results: up: mc = {up[DelMbc_index]:.3f}, down: mc = {down[DelMbc_index]:.3f}."
            )

        eigenvalue_variations.append((up, down))

    # in the case of the broad D**, additionally add the 1.5 sigma variations to account for wrong modelling
    # due to the non-consideration of their sizable widths in the theory prediction and in HAMMER:
    if model.Xc in ["D**0*", "D**1*"]:
        for i, eigenvalue in enumerate(eigenvalues):
            variation = np.zeros(len(cvs))
            variation[i] = np.sqrt(eigenvalue) * 1.5

            up = cvs + eigenvectors @ variation
            down = cvs - eigenvectors @ variation

            eigenvalue_variations.append((up, down))

    return eigenvalue_variations


class SavableAttributesObject:
    """A class for managing objects with savable attributes.

    This class provides a mechanism to register and store saving information for various attributes.

    Attributes:
        saving_info (dict): A dictionary to store saving-related information.
    """

    def __init__(self):
        """Initializes the SavableAttributesObject with an empty `saving_info` dictionary."""
        self.saving_info = {}

    def register_saving_info(self, saving_info: dict):
        """Registers the saving information for the object's attributes.

        Args:
            saving_info (dict): A dictionary containing information about how attributes should be saved.

        Example:
            >>> obj = SavableAttributesObject()
            >>> obj.register_saving_info({'attribute_name': 'save_path'})
        """
        self.saving_info = saving_info
