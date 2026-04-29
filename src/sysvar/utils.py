from __future__ import annotations

from os import path
from pathlib import Path
from yaml import safe_load
from typing import List, Optional
import numpy as np
import pandas as pd

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

    Arguments
    ---------
    covariance : numpy.ndarray
        The covariance matrix.

    Return
    ------
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


def load_covariance_matrix(config: dict, *, key: str = "cov_matrix") -> np.ndarray:
    """
    Load a covariance matrix from a config dict.

    Contract:
      - config MUST contain `key` (default: 'cov_matrix')
      - If config[key] is None: raise ValueError
      - If config[key] is a path (str/Path): load from file
      - If config[key] is array-like (list/tuple/np.ndarray): return np.ndarray
    """
    if key not in config:
        raise KeyError(f"Config must contain '{key}' key")

    cov = config[key]

    if cov is None:
        logging.info(f"'{key}' is None in config. No covariance matrix will be loaded.")
        return None

    # Path-like: load from file
    if isinstance(cov, (str, Path)):
        p = Path(cov)
        if not p.exists():
            raise ValueError(f"Covariance matrix file not found: {p}")

        logging.info(f"Loading covariance matrix from file: {p}")

        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".tsv":
            # return numpy array for consistency
            return np.loadtxt(p, delimiter='\t')

        # otherwise assume CSV-like numeric text
        return np.loadtxt(p, delimiter=",")

    # Array-like: use directly
    if isinstance(cov, (list, tuple, np.ndarray)):
        logging.info(f"Loading covariance matrix from config value '{key}'")
        return np.asarray(cov, dtype=float)

    raise ValueError(
        f"Unsupported type for '{key}': {type(cov)}. "
        "Expected None, path (str/Path), or array-like (list/tuple/np.ndarray)."
    )
