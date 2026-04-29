from __future__ import annotations

from typing import List, Iterable, Optional, Dict, Union, Any
from os import path
from warnings import warn
from pathlib import Path

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
    correction_source: str | Path | dict,
    MC_production: str | None = None,
    prefix: str | None = None,
    weightname: str | None = None,
    overwrite: bool = False,
    Nvar: int = 0,
):
    """
    Add correction weights (and optional toy variations) to a pandas DataFrame in-place.

    This function computes per-row correction weights using a correction object created
    from `correction_source` and writes them into `df` as a new column. The rows that
    receive a given correction value are determined by boolean query strings produced
    by `correction.build_queries(prefix)` and evaluated against `df` via `df.eval()`.

    If `Nvar > 0`, additional columns containing toy variations of the weights are
    added as well (one column per toy). The toy weights are produced via a `Variator`
    constructed from the correction.

    Parameters
    ----------
    df:
        The DataFrame to modify in-place.
    correction_source:
        Defines how to construct the correction:
          - `Path`: treated as a path to a CSV/TSV correction table.
          - `str`: either a correction identifier (YAML-based, legacy) or a path-like
            string to a CSV.
          - `dict`: a fully specified custom correction configuration.
    MC_production:
        MC production tag required for YAML-based corrections (legacy). Not used for
        CSV-based inputs. May be ignored depending on `correction_source`.
    prefix:
        Optional prefix used when building the dependent-variable column names used in
        the query strings (e.g. "trk" -> "trk_pt"). Passed through to
        `correction.build_queries(prefix)`.
    weightname:
        Base name of the weight column. The final column name is built by
        `correction._build_column_name(prefix, weightname)`.
    overwrite:
        If True and the target weight column already exists, it will be overwritten.
        If False and the column exists, the function logs a warning and does not
        modify the DataFrame.
    Nvar:
        Number of toy-variation columns to add. Must be a non-negative integer.
        If 0, only the central weight column is added. If > 0, columns named
        "{column_name}_var_{j}" for j in [0, Nvar-1] are added.

    Returns
    -------
    None
        The DataFrame is modified in-place.

    Raises
    ------
    ValueError
        If `Nvar` is negative, or if `correction_source` / `MC_production` do not form
        a valid combination for constructing a correction.
    Exception
        Propagates any exception raised by `create_correction_object`, `df.eval`,
        or by the correction/variator internals.

    Notes
    -----
    - The queries produced by `correction.build_queries(prefix)` are evaluated using
      `df.eval()`, so the DataFrame must contain the referenced columns.
    - For correctness, the correction's binning (number of central values / queries)
      must match the internal structure of the correction and the `Variator`.
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

    correction = create_correction_object(
        correction_source=correction_source,
        MC_production=MC_production,
    )
    column_name = correction._build_column_name(prefix, weightname)

    # Early skip: do not construct Variator and do not touch df
    if column_name in df.columns and not overwrite:
        logging.warning(
            "%s exists but it will not be overwritten. Skipping. "
            "If you want to overwrite set overwrite=True.",
            column_name,
        )
        return

    # Only build variator if we will actually write
    variator = Variator(correction, Nvar) if Nvar > 0 else None

    if column_name in df.columns and overwrite:
        logging.info("%s exists and will be overwritten", column_name)
    else:
        logging.info("%s does not exist. Adding it to dataframe", column_name)

    _add_weights(df, correction, prefix, column_name, variator)


def eigendecompose(
    df: DataFrame,
    settings: dict[str, Any],
    systematic_source: str | Path | dict,
    title: str | None = None,
    cov_matrix_path: str | Path | None = None,
    criterion: str = "max_differences",
    prc: float = 1e-4,
    max_variations: int | None = None,
    save_variations: bool = False,
    save_channel_covariance_matrices: bool = False,
    verbose: bool = True,
    seed: int = 8311311,
):
    """
    Run an eigendecomposition workflow and return the configured `EigenDecomposer`.

    This is a convenience wrapper around `EigenDecomposer` that:
      1) constructs an `EigenDecomposer`,
      2) generates template variations,
      3) applies the requested precision / variation limits,
      4) selects the important eigendimensions using `criterion`,
      5) optionally persists variations and/or per-channel covariance matrices.

    Parameters
    ----------
    df:
        Input DataFrame used by the decomposer (templates / channels / yields, as
        expected by `EigenDecomposer`).
    settings:
        Configuration dictionary consumed by `EigenDecomposer` (e.g. channel
        definitions, output paths, variables to use, etc.).
    systematic_source:
        Source used to build the underlying correction / systematic definition.
        Typically one of:
          - `str`: a correction/systematic identifier (e.g. YAML key; legacy),
          - `Path` or path-like `str`: a CSV file describing the correction,
          - `dict`: an in-memory correction configuration.
        The exact interpretation is delegated to `EigenDecomposer`.
    title (str | None, optional): 
        Custom title for CSV-based corrections. If not provided, will use the CSV filename.
    cov_matrix_path:
        Optional path to an explicit covariance matrix to use instead of building
        it from uncertainties. If provided, it is passed through to
        `EigenDecomposer`.
    criterion:
        Criterion used to select “important” eigendimensions. Must be understood
        by `EigenDecomposer.find_important_eigendimension_indices`.
        Default is `"max_differences"`.
    prc:
        Precision threshold used to determine how many eigendimensions to keep.
        Interpreted by the decomposer. Default is 1e-4.
    max_variations:
        Optional hard cap on the number of variations/eigendimensions to consider
        (after applying the precision criterion). If None, no cap is applied.
    save_variations:
        If True, calls `EigenDecomposer.save_template_variations()`. This performs
        file I/O to whatever output location the decomposer/settings define.
    save_channel_covariance_matrices:
        If True, calls `EigenDecomposer.save_channel_covariance_matrices()`. This
        performs file I/O.
    verbose:
        If True, enables verbose output/logging in `EigenDecomposer`.
    seed:
        Random seed forwarded to `EigenDecomposer` for reproducibility.

    Returns
    -------
    EigenDecomposer
        The initialized decomposer instance containing the decomposition results
        and selected eigendimensions.

    Notes
    -----
    This function has optional side effects (writing files) when `save_variations`
    and/or `save_channel_covariance_matrices` are enabled.
    """
    egd = EigenDecomposer(
        df=df,
        settings=settings,
        systematic_source=systematic_source,
        title=title,
        cov_matrix_path=cov_matrix_path,
        verbose=verbose,
        seed=seed,
    )

    egd.precision = prc
    egd.max_variations = max_variations

    egd.vary_templates()
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
