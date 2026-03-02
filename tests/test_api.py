import pytest
from pathlib import Path

from sysvar.api import (
    plot_analysis_corr_matrix,
    plot_cov_diff,
    plot_up_and_down_variations,
    plot_templates_relative_variations_in_grid,
    plot_correction_cov_and_corr,
    plot_correction_variations_in_grid,
    plot_correction_errors,
    add_weights_to_dataframe,
    eigendecompose,
)


from sysvar.eigendecomposer import EigenDecomposer
from sysvar.utils import read_yaml
import numpy as np

def _repo_root_from_test_file() -> Path:
    # tests/ is at <repo_root>/tests
    return Path(__file__).resolve().parents[1]


def _config_csv_path(csv_filename: str) -> Path:
    return _repo_root_from_test_file() / "configs" / "csv_configs" / csv_filename

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def dummy_eigendecomposer(toy_df, request):
    settings = read_yaml("study_setup", "sysvar_101")
    syst_effect = "charged_slow_pi"
    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)

    egd = EigenDecomposer(toy_df, settings, csv_path, verbose=False)
    egd.vary_templates()
    egd.precision = 1e-4
    egd.find_important_eigendimension_indices("max_differences")

    return egd


@pytest.mark.parametrize(
    "plot_method",
    [
        plot_analysis_corr_matrix,
        plot_cov_diff,
        plot_up_and_down_variations,
        plot_templates_relative_variations_in_grid,
        plot_correction_cov_and_corr,
        plot_correction_variations_in_grid,
        plot_correction_errors,
    ],
)
def test_visualize_functions_run_without_errors(plot_method, dummy_eigendecomposer):
    plot_method(dummy_eigendecomposer, save=False)

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def test_add_weights_to_dataframe_overwrite_and_variations(toy_df, request):
    df = toy_df.copy()
    column_name = "slow_pi_charged_weight"

    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)

    # Ensure the base column exists from fixture setup
    assert column_name in df.columns

    # Set a sentinel value and verify non-overwrite keeps it
    df[column_name] = 42.0
    add_weights_to_dataframe(
        df=df,
        #systematic="charged_slow_pi",
        #MC_production="sysvar_101",
        csv_path=csv_path,
        prefix="slow_pi",
        weightname="charged_weight",
        overwrite=False,
    )
    assert (df[column_name] == 42.0).all()

    # Overwrite and request two variation columns
    add_weights_to_dataframe(
        df=df,
        #systematic="charged_slow_pi",
        #MC_production="sysvar_101",
        csv_path=csv_path,
        prefix="slow_pi",
        weightname="charged_weight",
        overwrite=True,
        Nvar=2,
    )

    # Values should have been updated from the sentinel
    assert (df[column_name] != 42.0).any()

    # Variation columns should exist and be populated
    for j in range(2):
        vcol = f"{column_name}_var_{j}"
        assert vcol in df.columns
        assert df[vcol].notna().all()

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def test_add_weights_to_dataframe_negative_Nvar_raises(toy_df, request):
    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)
    df = toy_df.copy()
    with pytest.raises(ValueError):
        add_weights_to_dataframe(
            df=df,
            systematic="charged_slow_pi",
            MC_production="sysvar_101",
            csv_path=csv_path,
            prefix="slow_pi",
            weightname="charged_weight",
            Nvar=-1,
        )

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def test_eigendecompose_runs_and_sets_properties(toy_df, dummy_eigendecomposer, request):
    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)
    df = toy_df.copy()
    settings = read_yaml("study_setup", "sysvar_101")
    prc = 1e-3

    egd = eigendecompose(
        df=df,
        settings=settings,
        syst_effect="charged_slow_pi",
        csv_path=csv_path,
        criterion="max_differences",
        prc=prc,
        save_variations=False,
        save_channel_covariance_matrices=False,
        verbose=False,
    )

    assert isinstance(egd, EigenDecomposer)
    assert egd.precision == prc
    assert egd.seed == 8311311

    # Baseline comparisons against dummy instance (ensures consistent setup)
    assert egd.syst_effect == dummy_eigendecomposer.syst_effect
    assert set(egd.templates.keys()) == set(dummy_eigendecomposer.templates.keys())

    # important dims should be computed and be a boolean array
    assert hasattr(egd, "important_dims_indices")
    assert isinstance(egd.important_dims_indices, np.ndarray)
    assert egd.important_dims_indices.dtype == bool
    assert egd.N_important_dims == 3

    # templates should exist and have eigen information accessible
    assert len(egd.templates) > 0
    for name, t in egd.templates.items():
        # Access properties to ensure they are computable
        vals = t.eigen_values
        vecs = t.eigen_vectors
        vars_ = t.eigen_variations
        assert isinstance(vals, np.ndarray) and vals.size > 0
        assert isinstance(vecs, np.ndarray) and vecs.size > 0
        assert isinstance(vars_, np.ndarray) and vars_.size > 0

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def test_eigendecompose_max_variations(toy_df, request):
    df = toy_df.copy()
    settings = read_yaml("study_setup", "sysvar_101")
    max_vars = 1

    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)

    egd = eigendecompose(
        df=df,
        settings=settings,
        syst_effect="charged_slow_pi",
        csv_path=csv_path,
        criterion="max_differences",
        prc=1e-3,
        max_variations=max_vars,
        save_variations=False,
        save_channel_covariance_matrices=False,
        verbose=False,
    )

    assert egd.max_variations == max_vars
    assert egd.N_important_dims == max_vars

@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
    ]
)
def test_eigendecompose_invalid_criterion_raises(toy_df, request):
    settings = read_yaml("study_setup", "sysvar_101")
    csv_filename = request.param["csv"]
    csv_path = _config_csv_path(csv_filename)
    with pytest.raises(NotImplementedError):
        eigendecompose(
            df=toy_df,
            settings=settings,
            syst_effect="charged_slow_pi",
            csv_path=csv_path,
            criterion="not_a_method",
            verbose=False,
        )
