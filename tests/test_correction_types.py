import pytest
from os import path
from unittest.mock import patch
from sysvar.corrections import (
    create_correction_object,
    CorrectionBF,
    CorrectionPID,
    Correction1D,
    Correction2D,
    CustomCorrection,
    Correction1DFromCSV,
    Correction2DFromCSV,
    Correction3DFromCSV,
)
from sysvar.utils import _get_configs_dir, _get_parent_dir


CONFIG_DIR = path.join(_get_parent_dir(), _get_configs_dir("sysvar_101"))


@pytest.mark.parametrize(
    "correction_source, MC_production, expected_cls",
    [
        ("1D_correction", "sysvar_101", Correction1D),
        ("BF_101", "sysvar_101", CorrectionBF),
    ],
)
def test_create_correction_object_yaml_only_parametrized(
    correction_source, MC_production, expected_cls
):

    correction = create_correction_object(
        correction_source=correction_source, MC_production=MC_production
    )

    assert isinstance(correction, expected_cls)


def test_create_correction_object_custom_returns_custom_correction():
    # Minimal valid custom correction config (no explicit covariance)
    info = {
        "dependant_variable": "channel",
        "central_values": [0.95, 1.00, 1.05],
        "query_targets": ["ch_A", "ch_B", "ch_C"],
        "unit": "GeV",
        "title": "My custom correction",
        "uncertainties": {
            "fully_correlated": {"sys": [0.02, 0.03, 0.025]},
            "uncorrelated": {"stat": [0.01, 0.01, 0.01]},
        },
        "cov_matrix": None,
    }

    correction = create_correction_object(correction_source=info)

    assert isinstance(correction, CustomCorrection)


@pytest.mark.parametrize(
    "title, correction_source, cov_matrix_path, expected_cls",
    [
        (
            "1D_csv_correction",
            path.join(CONFIG_DIR, "1D_csv_correction.csv"),
            None,
            Correction1DFromCSV,
        ),
    ],
)
def test_create_correction_object_csv_only_parametrized(
    title, correction_source, cov_matrix_path, expected_cls
):

    correction = create_correction_object(
        correction_source=correction_source,
        cov_matrix_path=cov_matrix_path,
        title=title,
    )

    assert isinstance(correction, expected_cls)
