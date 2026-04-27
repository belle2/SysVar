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


CONFIG_DIR = path.join(_get_parent_dir(), _get_configs_dir("csv_configs"))


@pytest.mark.parametrize(
    "syst_effect, MC_prod, expected_cls",
    [
        ("charged_slow_pi", "MC15rd", Correction1D),
        ("neutral_slow_pi", "MC15rd", Correction1D),
        ("prompt_hadronic_Bp_BF", "MC15rd", CorrectionBF),
        ("prompt_hadronic_B0_BF", "MC15rd", CorrectionBF),
        ("double_charm_Bp_BF", "MC15rd", CorrectionBF),
        ("double_charm_B0_BF", "MC15rd", CorrectionBF),
        ("dst0_meson_BF", "MC15rd", CorrectionBF),
        ("tau_BF", "MC15rd", CorrectionBF),
        ("dstplus_meson_BF", "MC15rd", CorrectionBF),
        ("dstst_BF", "MC15rd", CorrectionBF),
    ],
)
def test_create_correction_object_yaml_only_parametrized(
    syst_effect, MC_prod, expected_cls
):

    correction = create_correction_object(syst_effect=syst_effect, MC_prod=MC_prod)

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
    }

    correction = create_correction_object(syst_effect=info)

    assert isinstance(correction, CustomCorrection)


@pytest.mark.parametrize(
    "title, csv_path, cov_matrix_path, expected_cls",
    [
        (
            "charged_slow_pi",
            path.join(CONFIG_DIR, "charged_slow_pi_correction.csv"),
            None,
            Correction1DFromCSV,
        ),
        (
            "neutral_pi_corr",
            path.join(CONFIG_DIR, "neutral_pi_corr.csv"),
            None,
            Correction2DFromCSV,
        ),
        (
            "Kshort_corr",
            path.join(CONFIG_DIR, "Kshort_corr.csv"),
            None,
            Correction3DFromCSV,
        ),
        (
            "fei_Bp",
            path.join(CONFIG_DIR, "fei_Bp_001.csv"),
            path.join(CONFIG_DIR, "Comb_Cov_RD_Bp_SigProb_0_001.npy"),
            Correction1DFromCSV,
        ),
    ],
)
def test_create_correction_object_csv_only_parametrized(
    title, csv_path, cov_matrix_path, expected_cls
):

    correction = create_correction_object(
        syst_effect=None,
        csv_path=csv_path,
        cov_matrix_path=cov_matrix_path,
        title=title,
    )

    assert isinstance(correction, expected_cls)
