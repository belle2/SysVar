from __future__ import annotations

from typing import List, Iterable, Optional, Dict, Union
from os import path

import numpy as np
from pandas import DataFrame, concat, read_csv

import matplotlib.pyplot as plt

from sysvar.variations import Variator
from sysvar.corrections import create_correction_object
from sysvar.eigendecomposer import EigenDecomposer, ExistingEigenVariationsSaver
from sysvar.visualize import CorrectionVisualizer, UncertaintyVisualizer
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
    Add weights to a DataFrame based on a correction object.

    Args:
        df (pd.DataFrame): The DataFrame to which weights should be added.
        correction: The correction object containing central values and queries.
        weightname (str): The name of the weight column to add.
        overwrite (bool, optional): Whether to overwrite the weight column if it already exists.
        Nvar: Number of variations to add to the dataframe. If 0, only the central value is added.

    Returns:
        None

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
        settings (Dict): Configuration settings for the `EigenDecomposer`.
        syst_effect (str): The systematic effect to consider in the decomposition.
        criterion (str): The criterion to determine important eigendimensions.
        prc (float, optional): The precision level for the decomposition process. Defaults to 1e-4.
        save_variations (bool, optional): If True, saves template variations during decomposition. Defaults to False.

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

    # Create an eigendecomposer object without any systematic effect
    ecth = ChannelTemplateHandler(df=df, settings=settings)
    ecth.save_nominal_templates(data=data)

    return ecth


def save_existing_eigenvariations(
    df: DataFrame, settings: Dict, syst_effect: str, verbose=True
):

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

    fig, ax = eigendecomposer_obj.plot_corr_matrix(save=save, filename=filename)
    return fig, ax


def plot_cov_diff(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:

    fig, ax = eigendecomposer_obj.plot_cov_diff(save=save, filename=filename)
    return fig, ax


def register_saving_info(eigendecomposer_obj: EigenDecomposer, saving_info: Dict):
    eigendecomposer_obj.register_saving_info(saving_info)


def plot_up_and_down_variations(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> List[tuple[plt.Figure, plt.Axes]]:

    figures = []

    for t in eigendecomposer_obj.templates.values():
        t.register_saving_info(eigendecomposer_obj.saving_info)
        fig, ax = t.plot_up_and_down_variations(save=save, filename=filename)
        figures.append((fig, ax))

    return figures


def plot_templates_relative_variations_in_grid(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> List[tuple[plt.Figure, plt.Axes]]:
    figures = []

    for t in eigendecomposer_obj.templates.values():
        t.register_saving_info(eigendecomposer_obj.saving_info)
        fig, ax = t.plot_relative_variations_in_grid(save=save, filename=filename)
        figures.append((fig, ax))

    return figures


def plot_correction_cov_and_corr(
    eigendecomposer_obj: EigenDecomposer,
    save: bool = False,
    filename: Union[None, str] = None,
) -> tuple[plt.Figure, plt.Axes]:

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

    eigendecomposer_obj.correction.register_saving_info(eigendecomposer_obj.saving_info)
    fig, ax = eigendecomposer_obj.correction.plot_error_comparison(
        save=save, filename=filename
    )
    return fig, ax
