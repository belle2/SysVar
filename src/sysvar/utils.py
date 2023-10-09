from os import path
from pathlib import Path
from yaml import safe_load
from typing import List

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
