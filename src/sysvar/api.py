from __future__ import annotations

from typing import List, Iterable, Optional, Dict, Union
from os import path

import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt

from sysvar.variations import Variator
from sysvar.corrections import create_correction_object
from sysvar.eigendecomposer import (
    EigenDecomposer,
    ExistingEigenVariationsSaver,
)
from sysvar.covariance_calculator import CovarianceCalculator
from sysvar.channel_template_handler import ChannelTemplateHandler

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


def add_weights_to_dataframe(
    df: DataFrame,
    systematic: str,
    MC_production: str,
    prefix: str,
    weightname: str,
    overwrite: bool = False,
    Nvar: int = 0,
):
    """
    Add weight columns to a DataFrame.

    This function augments `df` in-place by adding a central weight column whose
    name is constructed from `prefix` and `weightname`, and filling it using
    the configured correction for the given `systematic` and `MC_production`.
    If `Nvar > 0`, it also adds `Nvar` variation columns (named
    "{column_name}_var_{j}") and fills them from variations (toys) of the
    central values of the corrections. A dedicated `Variator` object is used
    internally to generate the toys.

    Parameters:
        df (DataFrame): pandas DataFrame to be augmented (modified in-place).
        systematic (str): Name of the systematic/correction to apply.
        MC_production (str): MC production tag used to locate the correction.
        prefix (str): Prefix used when building the weight column name.
        weightname (str): Base name of the weight column to add.
        overwrite (bool, optional): If True, overwrite an existing column with
            the same name. Defaults to False.
        Nvar (int, optional): Number of variation columns to add. Must be a
            non-negative integer. If 0, only the central value column is added.
            Defaults to 0.

    Returns:
        None: The DataFrame `df` is modified in-place.

    Raises:
        ValueError: If `Nvar` is negative.
    """

    if Nvar < 0:
        raise ValueError("Nvar must be a positive integer")

    def _add_weights(df, correction, prefix, column_name, variator=None):

        df.loc[:, column_name] = 1.0
        if variator is not None:
            variation_columns = [f"{column_name}_var_{j}" for j in range(variator.Nvar)]
            df.loc[:, variation_columns] = 1.0

        for i, (v, q) in enumerate(
            zip(correction.central_values, correction.build_queries(prefix))
        ):
            mask = df.eval(q)
            df.loc[mask, column_name] = v

            if variator is not None:
                df.loc[mask, variation_columns] = variator.variations[:, i]

    correction = create_correction_object(systematic, MC_production)
    column_name = correction._build_column_name(prefix, weightname)

    variator = Variator(correction, Nvar) if Nvar > 0 else None

    if column_name in df.columns and overwrite:
        logging.info("%s exists but it will be overwriten", column_name)

        _add_weights(df, correction, prefix, column_name, variator)

    elif column_name in df.columns and not overwrite:

        logging.warning(
            "%s exists but it not will be ovewritten. Skipping. No weights are added. If you want to change this behaviour set the overwrite argument to True",
            column_name,
        )
    elif column_name not in df.columns:
        logging.info("%s does not exist. Adding it to dataframe", column_name)
        _add_weights(df, correction, prefix, column_name, variator)


def eigendecompose(
    df: DataFrame,
    settings: Dict,
    syst_effect: str,
    criterion: str = "max_differences",
    prc: float = 1e-4,
    save_variations: bool = False,
    save_channel_covariance_matrices: bool = False,
    verbose: bool = True,
):
    """
    Performs eigendecomposition on the input DataFrame based on specified settings,
    systematic effect, and criterion, and returns the resulting `EigenDecomposer` object.

    This function initializes an `EigenDecomposer` instance using the provided
    DataFrame, settings, and systematic effect. It then applies a precision level
    to the decomposition and identifies important eigendimension indices based on
    the specified criterion. Optionally, it saves template variations.

    Args:
        df (DataFrame): The input data to be decomposed.
        settings (Dict): Configuration settings for the decomposition.
        syst_effect (str): The systematic effect to consider.
        criterion (str, optional): Criterion for selecting important eigendimensions.
            Defaults to "max_differences".
        prc (float, optional): Precision level for the decomposition. Defaults to 1e-4.
        save_variations (bool, optional): If True, saves template variations. Defaults to False.
        save_channel_covariance_matrices (bool, optional): If True, saves covariance matrices per channel. Defaults to False.
        verbose (bool, optional): If True, prints additional information during execution. Defaults to True.

    Returns:
        EigenDecomposer: An instance of the `EigenDecomposer` class containing the decomposition results.
    """

    egd = EigenDecomposer(df, settings, syst_effect, verbose=verbose)
    egd.vary_templates()
    egd.precision = prc
    egd.find_important_eigendimension_indices(criterion)

    if save_variations:
        egd.save_template_variations()
    if save_channel_covariance_matrices:
        egd.save_channel_covariance_matrices()

    return egd


def save_nominal_templates(df: DataFrame, settings: Dict, data=None):
    """Save nominal templates for an MC dataset.

    Write nominal templates for a Monte Carlo (MC) dataset (and optional
    experimental data) to the output file configured in `settings`. Only ROOT
    (.root) files are currently supported. Channels, templates, signal
    extraction variables, binning, and other required configuration are read
    from the `settings` dictionary. The produced file structure is compatible
    with cabinetry and can be used to build a pyhf model including systematic
    uncertainties.

    The function creates or recreates the configured output file on disk and
    therefore will overwrite any existing file at that location. Because the
    file is recreated, eigenvariation histograms saved before calling this
    function would be lost; call this function before saving eigenvariations
    for systematics.

    Parameters:
        df (DataFrame): MC dataset containing template information used to
            build the nominal templates (e.g., event records or pre-binned
            contents). This object is read but not modified by the function.
        settings (Dict): Configuration dictionary. Must include the output
            filename (currently a .root path) and the definitions for channels,
            templates, signal-extraction variables and binning required to
            produce the histograms.
        data (optional): Experimental (observed) dataset to be histogrammed and
            included in the output using the same channels, variables and
            binning as the MC templates. If None, no observed-data histograms
            are written.

    Returns:
        ChannelTemplateHandler: The handler object used to save the nominal
            templates. This object can be returned for later use inspect
            saved templates).
    """

    # Create an eigendecomposer object without any systematic effect
    ecth = ChannelTemplateHandler(df=df, settings=settings)
    ecth.save_nominal_templates(data=data)

    return ecth


def save_existing_eigenvariations(
    df: DataFrame, settings: Dict, syst_effect: str, verbose=True
):
    """Save existing eigenvariations for a specified systematic effect.

    This function complements the nominal-template saving step:
    nominal templates should already be present in the configured output file
    before calling this function.

    The saver will read variations from `df` and write eigenvariation histograms
    into the same ROOT file structure expected by cabinetry/pyhf so that the model
    can later be built including these systematic eigenvariations.
    Instead of using the nominal weights for the histogram filling, the nominal weight
    for the syst_effect is replaced by the variations present in `df`.
    The number of variations that should be present in `df` is read from the settings
    dictionary in the systematics part.

    Args:
        df (DataFrame): The dataset to extract variations from.
        settings (Dict): Configuration settings for eigenvariation saving.
        syst_effect (str): The systematic effect to save variations for.
        verbose (bool, optional): If True, enables verbose logging. Defaults to True.

    Returns:
        None
    """
    ees = ExistingEigenVariationsSaver(df, settings)
    ees.syst_effect = syst_effect
    ees.save_existing_eigenvariations(verbose=verbose)


def calculate_covariance_matrix(
    df: DataFrame,
    settings: Dict,
    syst_effect: str | Dict,
    binning: Dict,
    channels: List,
    input_cov: np.ndarray = None,
    save_cov: bool = False,
):
    """
    Calculate the covariance matrix for a given dataset.

    This function computes a covariance matrix based on the input data, configuration settings, and systematic effects. It provides support for pre-defined systematics or custom-defined ones and allows the user to specify binning and channels. Optionally, it can save the covariance matrix to a file.

    Args:
        df (DataFrame): The input data to calculate the covariance matrix from.
        settings (Dict): Configuration settings, same as for the `EigenDecomposer`.
        syst_effect (str | Dict): The name of the systematic effect to consider for the covariance matrix. For systematics from YAML files the name is enough. If this is a custom systematic then a dictionary with for the custom systematic is expected similarly to the dictionary necessary for the custom correction object in the eigendecomposition.
        binning (Dict): Binning information for the covariance matrix. Keys should be the variable names present in the df and values lists of bin edges.
        channels (List): List of channels to consider for the covariance matrix.
        save_cov (bool, optional): If True, saves the covariance matrix. The path should be read from the settings dictionary. Defaults to False.

    Returns:
        the covariance matrix from the covariance matrix calculator
    """

    cc = CovarianceCalculator(df, settings, syst_effect, binning, channels, input_cov)
    cc.vary_templates()
    if save_cov:
        cc.save_covariance()
    return cc.cov


def plot_analysis_corr_matrix(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the correlation matrix of an eigendecomposition analysis.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing correlation data.
        save (bool, optional): If True, saves the plot to file. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    fig, ax = eigendecomposer_obj.plot_corr_matrix(save=save, filename=filename)
    return fig, ax


def plot_cov_diff(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the normalized covariance difference between original and eigendecomposed covariance matrix for an initial truncation guess.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing covariance data.
        save (bool, optional): If True, saves the plot. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    fig, ax = eigendecomposer_obj.plot_cov_diff(save=save, filename=filename)
    return fig, ax


def register_saving_info(eigendecomposer_obj: EigenDecomposer, saving_info: Dict):
    """
    Register saving information in the eigendecomposer object.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object to update.
        saving_info (Dict): Dictionary containing saving parameters.

    Returns:
        None
    """
    eigendecomposer_obj.register_saving_info(saving_info)


def plot_up_and_down_variations(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> List[tuple[plt.Figure, plt.Axes]]:
    """
    Plot up/down variations for each template in the decomposition.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing templates.
        save (bool, optional): If True, saves the plots. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    figures = []

    for t_name, t_obj in eigendecomposer_obj.templates.items():
        t_obj.register_saving_info(eigendecomposer_obj.saving_info)
        fig, ax = t_obj.plot_up_and_down_variations(
            title=t_name, save=save, filename=filename
        )
        figures.append((fig, ax))

    return figures


def plot_templates_relative_variations_in_grid(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> List[tuple[plt.Figure, plt.Axes]]:
    """
    Plot relative template variations in a grid layout.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing templates.
        save (bool, optional): If True, saves the plots. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    figures = []

    for t_name, t_obj in eigendecomposer_obj.templates.items():
        t_obj.register_saving_info(eigendecomposer_obj.saving_info)
        fig, ax = t_obj.plot_relative_variations_in_grid(
            title=t_name, save=save, filename=filename
        )
        figures.append((fig, ax))

    return figures


def plot_correction_cov_and_corr(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot correction covariance and correlation matrices.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing correction data.
        save (bool, optional): If True, saves the plot. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    eigendecomposer_obj.variator.register_saving_info(eigendecomposer_obj.saving_info)
    fig, ax = eigendecomposer_obj.variator.plot_cov_and_corr(
        save=save, filename=filename
    )
    return fig, ax


def plot_correction_variations_in_grid(
    eigendecomposer_obj: EigenDecomposer,
    nbins=21,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot correction variations in a grid layout.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing correction variations.
        nbins (int, optional): Number of bins to use in the grid plot. Defaults to 21.
        save (bool, optional): If True, saves the plots. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    eigendecomposer_obj.variator.register_saving_info(eigendecomposer_obj.saving_info)
    fig, ax = eigendecomposer_obj.variator.plot_relative_variations_in_grid(
        nbins=nbins, save=save, filename=filename
    )
    return fig, ax


def plot_correction_errors(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot correction error comparisons.

    Args:
        eigendecomposer_obj (EigenDecomposer): The decomposition object containing correction information.
        save (bool, optional): If True, saves the plot. Defaults to False.
        filename (str, optional): Output file name if saving. Defaults to None.

    Returns:
        None
    """

    eigendecomposer_obj.correction.register_saving_info(eigendecomposer_obj.saving_info)
    fig, ax = eigendecomposer_obj.correction.plot_error_comparison(
        save=save, filename=filename
    )
    return fig, ax
