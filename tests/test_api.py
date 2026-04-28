# tests/test_api_surface_smoke.py
from __future__ import annotations

import inspect

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Adjust this import to the module that contains the API you pasted
from sysvar.api import (
    add_weights_to_dataframe,
    eigendecompose,
    plot_analysis_corr_matrix,
    plot_cov_diff,
    register_saving_info,
    plot_up_and_down_variations,
    plot_templates_relative_variations_in_grid,
    plot_correction_cov_and_corr,
    plot_correction_variations_in_grid,
    plot_correction_errors,
)


# -----------------------------------------------------------------------------
# Signature stability (argument names & defaults)
# -----------------------------------------------------------------------------


def test_add_weights_to_dataframe_signature_is_stable():
    sig = inspect.signature(add_weights_to_dataframe)
    assert list(sig.parameters.keys()) == [
        "df",
        "correction_source",
        "MC_production",
        "prefix",
        "weightname",
        "overwrite",
        "Nvar",
    ]
    p = sig.parameters
    assert p["MC_production"].default is None
    assert p["prefix"].default is None
    assert p["weightname"].default is None
    assert p["overwrite"].default is False
    assert p["Nvar"].default == 0


def test_eigendecompose_signature_is_stable():
    sig = inspect.signature(eigendecompose)
    assert list(sig.parameters.keys()) == [
        "df",
        "settings",
        "systematic_source",
        "cov_matrix_path",
        "criterion",
        "prc",
        "max_variations",
        "save_variations",
        "save_channel_covariance_matrices",
        "verbose",
        "seed",
    ]
    p = sig.parameters
    assert p["cov_matrix_path"].default is None
    assert p["criterion"].default == "max_differences"
    assert p["prc"].default == 1e-4
    assert p["max_variations"].default is None
    assert p["save_variations"].default is False
    assert p["save_channel_covariance_matrices"].default is False
    assert p["verbose"].default is True
    assert p["seed"].default == 8311311


@pytest.mark.parametrize(
    "fn, expected_params",
    [
        (plot_analysis_corr_matrix, ["eigendecomposer_obj", "save", "filename"]),
        (plot_cov_diff, ["eigendecomposer_obj", "save", "filename"]),
        (register_saving_info, ["eigendecomposer_obj", "saving_info"]),
        (plot_up_and_down_variations, ["eigendecomposer_obj", "save", "filename"]),
        (
            plot_templates_relative_variations_in_grid,
            ["eigendecomposer_obj", "save", "filename"],
        ),
        (plot_correction_cov_and_corr, ["eigendecomposer_obj", "save", "filename"]),
        (
            plot_correction_variations_in_grid,
            ["eigendecomposer_obj", "nbins", "save", "filename"],
        ),
        (plot_correction_errors, ["eigendecomposer_obj", "save", "filename"]),
    ],
)
def test_plot_wrapper_signatures_are_stable(fn, expected_params):
    sig = inspect.signature(fn)
    assert list(sig.parameters.keys()) == expected_params


# -----------------------------------------------------------------------------
# Plotting smoke tests (ensure wrappers call through and return fig/ax)
# -----------------------------------------------------------------------------


class _DummyVariator:
    def __init__(self):
        self._saving_info = None

    def register_saving_info(self, saving_info):
        self._saving_info = saving_info

    def plot_cov_and_corr(self, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax

    def plot_relative_variations_in_grid(self, nbins=21, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax


class _DummyCorrection:
    def __init__(self):
        self._saving_info = None

    def register_saving_info(self, saving_info):
        self._saving_info = saving_info

    def plot_error_comparison(self, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax


class _DummyTemplate:
    def __init__(self):
        self._saving_info = None

    def register_saving_info(self, saving_info):
        self._saving_info = saving_info

    def plot_up_and_down_variations(self, title=None, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax

    def plot_relative_variations_in_grid(self, title=None, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax


class _DummyEigenDecomposer:
    def __init__(self):
        self.saving_info = {"out": "dummy.root"}
        self.variator = _DummyVariator()
        self.correction = _DummyCorrection()
        self.templates = {"t1": _DummyTemplate(), "t2": _DummyTemplate()}

    def register_saving_info(self, saving_info):
        self.saving_info = saving_info

    def plot_corr_matrix(self, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax

    def plot_cov_diff(self, save=False, filename=None):
        fig, ax = plt.subplots()
        return fig, ax


def test_plot_analysis_corr_matrix_smoke():
    egd = _DummyEigenDecomposer()
    fig, ax = plot_analysis_corr_matrix(egd, save=False, filename=None)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_plot_cov_diff_smoke():
    egd = _DummyEigenDecomposer()
    fig, ax = plot_cov_diff(egd, save=False, filename=None)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_register_saving_info_smoke():
    egd = _DummyEigenDecomposer()
    register_saving_info(egd, {"out": "new.root"})
    assert egd.saving_info["out"] == "new.root"


def test_plot_up_and_down_variations_smoke():
    egd = _DummyEigenDecomposer()
    figures = plot_up_and_down_variations(egd, save=False, filename=None)
    assert isinstance(figures, list)
    assert len(figures) == len(egd.templates)
    for fig, ax in figures:
        assert fig is not None and ax is not None
        plt.close(fig)


def test_plot_templates_relative_variations_in_grid_smoke():
    egd = _DummyEigenDecomposer()
    figures = plot_templates_relative_variations_in_grid(egd, save=False, filename=None)
    assert isinstance(figures, list)
    assert len(figures) == len(egd.templates)
    for fig, ax in figures:
        assert fig is not None and ax is not None
        plt.close(fig)


def test_plot_correction_cov_and_corr_smoke():
    egd = _DummyEigenDecomposer()
    fig, ax = plot_correction_cov_and_corr(egd, save=False, filename=None)
    assert fig is not None and ax is not None
    plt.close(fig)
    # Wrapper should forward saving_info to variator
    assert egd.variator._saving_info == egd.saving_info


def test_plot_correction_variations_in_grid_smoke():
    egd = _DummyEigenDecomposer()
    fig, ax = plot_correction_variations_in_grid(
        egd, nbins=15, save=False, filename=None
    )
    assert fig is not None and ax is not None
    plt.close(fig)
    assert egd.variator._saving_info == egd.saving_info


def test_plot_correction_errors_smoke():
    egd = _DummyEigenDecomposer()
    fig, ax = plot_correction_errors(egd, save=False, filename=None)
    assert fig is not None and ax is not None
    plt.close(fig)
    # Wrapper should forward saving_info to correction
    assert egd.correction._saving_info == egd.saving_info


class _SpyCorrection:
    def __init__(self):
        self.central_values = [1.0]
        self.N = 1
        self.seen_prefix_in_build_queries = None
        self.seen_prefix_weightname_in_build_column_name = None

    def build_queries(self, prefix=None):
        self.seen_prefix_in_build_queries = prefix
        return ["x == x"]  # always true

    def _build_column_name(self, prefix, weightname):
        self.seen_prefix_weightname_in_build_column_name = (prefix, weightname)
        if prefix and weightname:
            return f"{prefix}_{weightname}"
        return "weight"


class _SpyVariator:
    last_init_args = None  # class-level storage for last call

    def __init__(self, correction, Nvar: int):
        type(self).last_init_args = (correction, Nvar)
        self.Nvar = Nvar
        self.variations = np.ones((Nvar, correction.N))


def test_add_weights_forwards_args_to_create_correction_object(monkeypatch):
    import sysvar.api as api_mod

    seen = {}

    def _spy_factory(**kwargs):
        seen.update(kwargs)
        return _SpyCorrection()

    monkeypatch.setattr(api_mod, "create_correction_object", _spy_factory)
    monkeypatch.setattr(api_mod, "Variator", _SpyVariator)

    df = pd.DataFrame({"x": [0, 1, 2]})
    src = Path("some.csv")

    add_weights_to_dataframe(
        df=df,
        correction_source=src,
        MC_production=None,
        prefix="pfx",
        weightname="w",
        overwrite=True,
        Nvar=0,
    )

    # verify exact kw forwarding to the factory
    assert seen == {"correction_source": src, "MC_production": None}


def test_add_weights_calls_build_column_name_with_prefix_and_weightname(monkeypatch):
    import sysvar.api as api_mod

    corr = _SpyCorrection()

    monkeypatch.setattr(api_mod, "create_correction_object", lambda **kwargs: corr)
    monkeypatch.setattr(api_mod, "Variator", _SpyVariator)

    df = pd.DataFrame({"x": [0, 1, 2]})
    add_weights_to_dataframe(
        df=df,
        correction_source="charged_slow_pi",
        MC_production="sysvar_101",
        prefix="slow_pi",
        weightname="charged_weight",
        overwrite=True,
        Nvar=0,
    )

    assert corr.seen_prefix_weightname_in_build_column_name == (
        "slow_pi",
        "charged_weight",
    )
    assert "slow_pi_charged_weight" in df.columns


def test_add_weights_calls_build_queries_with_prefix(monkeypatch):
    import sysvar.api as api_mod

    corr = _SpyCorrection()

    monkeypatch.setattr(api_mod, "create_correction_object", lambda **kwargs: corr)
    monkeypatch.setattr(api_mod, "Variator", _SpyVariator)

    df = pd.DataFrame({"x": [0, 1, 2]})
    add_weights_to_dataframe(
        df=df,
        correction_source={"some": "config"},
        prefix="trk",
        weightname="w",
        overwrite=True,
        Nvar=0,
    )

    assert corr.seen_prefix_in_build_queries == "trk"


def test_add_weights_constructs_variator_only_when_Nvar_positive(monkeypatch):
    import sysvar.api as api_mod

    corr = _SpyCorrection()

    monkeypatch.setattr(api_mod, "create_correction_object", lambda **kwargs: corr)
    monkeypatch.setattr(api_mod, "Variator", _SpyVariator)

    df = pd.DataFrame({"x": [0, 1]})

    # Nvar == 0: variator should not be created
    _SpyVariator.last_init_args = None
    add_weights_to_dataframe(
        df=df,
        correction_source="x",
        overwrite=True,
        Nvar=0,
    )
    assert _SpyVariator.last_init_args is None

    # Nvar > 0: variator should be created with (correction, Nvar)
    _SpyVariator.last_init_args = None
    add_weights_to_dataframe(
        df=df,
        correction_source="x",
        prefix="p",
        weightname="w",
        overwrite=True,
        Nvar=3,
    )
    assert _SpyVariator.last_init_args == (corr, 3)
