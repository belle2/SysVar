"""
Tests for the matrix utility functions:
  - corr2cov: compute covariance matrix from correlation matrix + variance vector
  - cov2corr: compute correlation matrix from covariance matrix
"""

import numpy as np
import pytest

# Adjust the import path to wherever these functions live in your package
from sysvar.utils import corr2cov, cov2corr


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_2x2():
    """2×2 identity correlation matrix and unit variances."""
    corr = np.eye(2)
    var = np.ones(2)
    return corr, var


@pytest.fixture
def simple_corr_and_var():
    """A symmetric positive-definite 3×3 correlation matrix with known variances."""
    corr = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )
    var = np.array([4.0, 9.0, 16.0])  # std devs: 2, 3, 4
    return corr, var


@pytest.fixture
def simple_cov(simple_corr_and_var):
    """Covariance matrix derived from simple_corr_and_var (used in cov2corr tests)."""
    corr, var = simple_corr_and_var
    return corr2cov(corr, var)


# ===========================================================================
# corr2cov tests
# ===========================================================================


class TestCorr2Cov:

    def test_identity_corr_with_unit_var_returns_identity(self, identity_2x2):
        """corr=I, var=1 => cov=I."""
        corr, var = identity_2x2
        cov = corr2cov(corr, var)
        np.testing.assert_array_almost_equal(cov, np.eye(2))

    def test_identity_corr_scales_correctly(self):
        """corr=I, var=[a,b] => cov=diag(a^2, b^2)."""
        corr = np.eye(3)
        var = np.array([2.0, 3.0, 4.0])
        cov = corr2cov(corr, var)
        expected = np.diag(var**2)
        np.testing.assert_array_almost_equal(cov, expected)

    def test_known_values(self, simple_corr_and_var):
        """Manually verify a 3×3 result: cov_ij = var_i * corr_ij * var_j."""
        corr, var = simple_corr_and_var
        cov = corr2cov(corr, var)

        # Check a few off-diagonal entries explicitly
        # cov[0,1] = var[0] * corr[0,1] * var[1] = 2*0.5*3 = 3.0
        assert pytest.approx(cov[0, 1]) == var[0] * corr[0, 1] * var[1]
        assert pytest.approx(cov[1, 2]) == var[1] * corr[1, 2] * var[2]
        assert pytest.approx(cov[0, 2]) == var[0] * corr[0, 2] * var[2]

    def test_diagonal_of_result_equals_var_squared(self, simple_corr_and_var):
        """Diagonal of cov should equal var^2 when corr diagonal is 1."""
        corr, var = simple_corr_and_var
        cov = corr2cov(corr, var)
        np.testing.assert_array_almost_equal(np.diag(cov), var**2)

    def test_output_is_symmetric(self, simple_corr_and_var):
        """Result must be symmetric."""
        corr, var = simple_corr_and_var
        cov = corr2cov(corr, var)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_output_shape(self, simple_corr_and_var):
        """Output shape must match the input correlation matrix shape."""
        corr, var = simple_corr_and_var
        cov = corr2cov(corr, var)
        assert cov.shape == corr.shape

    def test_zero_variance_produces_zero_row_col(self):
        """A zero variance entry should zero out its row and column."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        var = np.array([0.0, 3.0])
        cov = corr2cov(corr, var)
        np.testing.assert_array_equal(cov[0, :], 0.0)
        np.testing.assert_array_equal(cov[:, 0], 0.0)

    def test_returns_ndarray(self, simple_corr_and_var):
        corr, var = simple_corr_and_var
        result = corr2cov(corr, var)
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_roundtrip_with_cov2corr(self, n):
        """corr2cov followed by cov2corr should recover the original correlation matrix."""
        rng = np.random.default_rng(42)
        # Build a valid SPD correlation matrix via random orthogonal matrix
        A = rng.standard_normal((n, n))
        raw = A @ A.T + n * np.eye(n)
        d = np.sqrt(np.diag(raw))
        corr = raw / np.outer(d, d)
        var = rng.uniform(0.5, 5.0, n)

        cov = corr2cov(corr, var)
        recovered_corr = cov2corr(cov)
        np.testing.assert_array_almost_equal(recovered_corr, corr, decimal=10)


# ===========================================================================
# cov2corr tests
# ===========================================================================


class TestCov2Corr:

    def test_identity_returns_identity(self):
        """cov=I => corr=I."""
        cov = np.eye(3)
        corr = cov2corr(cov)
        np.testing.assert_array_almost_equal(corr, np.eye(3))

    def test_diagonal_values_are_one(self, simple_cov):
        """Diagonal entries of the correlation matrix must all be 1."""
        corr = cov2corr(simple_cov)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(corr.shape[0]))

    def test_output_is_symmetric(self, simple_cov):
        """Result must be symmetric."""
        corr = cov2corr(simple_cov)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_values_in_minus_one_to_one(self, simple_cov):
        """All correlation values must be in [-1, 1]."""
        corr = cov2corr(simple_cov)
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)

    def test_output_shape_matches_input(self, simple_cov):
        """Output shape must equal input shape."""
        corr = cov2corr(simple_cov)
        assert corr.shape == simple_cov.shape

    def test_returns_ndarray(self, simple_cov):
        result = cov2corr(simple_cov)
        assert isinstance(result, np.ndarray)

    def test_known_off_diagonal_values(self, simple_corr_and_var, simple_cov):
        """Recovered off-diagonals should match the original correlation entries."""
        original_corr, _ = simple_corr_and_var
        recovered = cov2corr(simple_cov)
        np.testing.assert_array_almost_equal(recovered, original_corr, decimal=10)

    def test_zero_covariance_entries_set_to_zero(self):
        """Where covariance is zero, the resulting correlation should also be zero."""
        cov = np.array(
            [
                [4.0, 0.0],
                [0.0, 9.0],
            ]
        )
        corr = cov2corr(cov)
        assert corr[0, 1] == 0.0
        assert corr[1, 0] == 0.0

    def test_scaled_identity(self):
        """Diagonal covariance matrix => identity correlation matrix."""
        cov = np.diag([1.0, 4.0, 9.0, 16.0])
        corr = cov2corr(cov)
        np.testing.assert_array_almost_equal(corr, np.eye(4))

    def test_output_is_real(self, simple_cov):
        """Result must be real-valued (no imaginary component)."""
        corr = cov2corr(simple_cov)
        assert np.isrealobj(corr)

    @pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
    def test_scale_invariance(self, scale):
        """Correlation matrix must be invariant to uniform scaling of the covariance."""
        cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        corr_original = cov2corr(cov)
        corr_scaled = cov2corr(cov * scale)
        np.testing.assert_array_almost_equal(corr_scaled, corr_original, decimal=10)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_roundtrip_with_corr2cov(self, n):
        """cov2corr followed by corr2cov should recover the original covariance matrix."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((n, n))
        cov = A @ A.T + n * np.eye(n)

        corr = cov2corr(cov)
        var = np.sqrt(np.diag(cov))
        recovered_cov = corr2cov(corr, var)
        np.testing.assert_array_almost_equal(recovered_cov, cov, decimal=10)


"""
Tests for the load_covariance_matrix utility function.

Covers:
  - Loading directly from a config dict (key_matrix)
  - Loading from a .npy file (key_path)
  - Loading from a .tsv file (key_path)
  - Loading from a .csv file (key_path, default delimiter)
  - Returns None when neither key is present
  - Logging behaviour (info vs warning)
"""

import logging
import numpy as np
import pandas as pd
import pytest

# Adjust import to wherever the function lives in your package
from sysvar.utils import load_covariance_matrix


# ---------------------------------------------------------------------------
# Shared reference data
# ---------------------------------------------------------------------------

REFERENCE_MATRIX = np.array(
    [
        [4.0, 1.0, 0.5],
        [1.0, 9.0, 2.0],
        [0.5, 2.0, 16.0],
    ]
)


# ===========================================================================
# Loading from the config dict directly (key_matrix branch)
# ===========================================================================


class TestLoadFromConfigKey:

    def test_returns_ndarray_when_key_matrix_present(self):
        config = {"my_cov": REFERENCE_MATRIX.tolist()}
        result = load_covariance_matrix(config, key_matrix="my_cov")
        assert isinstance(result, np.ndarray)

    def test_values_match_when_loaded_from_config(self):
        config = {"my_cov": REFERENCE_MATRIX.tolist()}
        result = load_covariance_matrix(config, key_matrix="my_cov")
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_key_matrix_takes_priority_over_key_path(self, tmp_path):
        """key_matrix branch must be preferred when both keys are present."""
        npy_file = tmp_path / "other.npy"
        np.save(npy_file, np.eye(3))

        config = {
            "my_cov": REFERENCE_MATRIX.tolist(),
            "cov_matrix_path": str(npy_file),
        }
        result = load_covariance_matrix(config, key_matrix="my_cov")
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_ndarray_input_is_preserved(self):
        """Passing an ndarray directly in the config should work too."""
        config = {"my_cov": REFERENCE_MATRIX}
        result = load_covariance_matrix(config, key_matrix="my_cov")
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_logs_info_when_loading_from_key_matrix(self, caplog):
        config = {"my_cov": REFERENCE_MATRIX.tolist()}
        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config, key_matrix="my_cov")
        assert any("my_cov" in msg for msg in caplog.messages)


# ===========================================================================
# Loading from a .npy file (key_path branch)
# ===========================================================================


class TestLoadFromNpyFile:

    def test_returns_ndarray(self, tmp_path):
        npy_file = tmp_path / "cov.npy"
        np.save(npy_file, REFERENCE_MATRIX)
        config = {"cov_matrix_path": str(npy_file)}
        result = load_covariance_matrix(config)
        assert isinstance(result, np.ndarray)

    def test_values_match(self, tmp_path):
        npy_file = tmp_path / "cov.npy"
        np.save(npy_file, REFERENCE_MATRIX)
        config = {"cov_matrix_path": str(npy_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_custom_key_path(self, tmp_path):
        npy_file = tmp_path / "cov.npy"
        np.save(npy_file, REFERENCE_MATRIX)
        config = {"custom_path_key": str(npy_file)}
        result = load_covariance_matrix(config, key_path="custom_path_key")
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_logs_info_when_loading_from_npy(self, tmp_path, caplog):
        npy_file = tmp_path / "cov.npy"
        np.save(npy_file, REFERENCE_MATRIX)
        config = {"cov_matrix_path": str(npy_file)}
        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config)
        assert any(str(npy_file) in msg for msg in caplog.messages)


# ===========================================================================
# Loading from a .tsv file (key_path branch)
# ===========================================================================


class TestLoadFromTsvFile:

    def test_returns_dataframe(self, tmp_path):
        """TSV branch returns a DataFrame (as per the implementation)."""
        tsv_file = tmp_path / "cov.tsv"
        pd.DataFrame(REFERENCE_MATRIX).to_csv(
            tsv_file, sep="\t", index=False, header=True
        )
        config = {"cov_matrix_path": str(tsv_file)}
        result = load_covariance_matrix(config)
        assert isinstance(result, pd.DataFrame)

    def test_values_match(self, tmp_path):
        tsv_file = tmp_path / "cov.tsv"
        pd.DataFrame(REFERENCE_MATRIX).to_csv(
            tsv_file, sep="\t", index=False, header=True
        )
        config = {"cov_matrix_path": str(tsv_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result.values, REFERENCE_MATRIX)

    def test_shape_is_correct(self, tmp_path):
        tsv_file = tmp_path / "cov.tsv"
        pd.DataFrame(REFERENCE_MATRIX).to_csv(
            tsv_file, sep="\t", index=False, header=True
        )
        config = {"cov_matrix_path": str(tsv_file)}
        result = load_covariance_matrix(config)
        assert result.shape == REFERENCE_MATRIX.shape

    def test_logs_info_when_loading_from_tsv(self, tmp_path, caplog):
        tsv_file = tmp_path / "cov.tsv"
        pd.DataFrame(REFERENCE_MATRIX).to_csv(
            tsv_file, sep="\t", index=False, header=True
        )
        config = {"cov_matrix_path": str(tsv_file)}
        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config)
        assert any(str(tsv_file) in msg for msg in caplog.messages)


# ===========================================================================
# Loading from a CSV file (key_path branch, default / fallback)
# ===========================================================================


class TestLoadFromCsvFile:

    def test_returns_ndarray(self, tmp_path):
        csv_file = tmp_path / "cov.csv"
        np.savetxt(csv_file, REFERENCE_MATRIX, delimiter=",")
        config = {"cov_matrix_path": str(csv_file)}
        result = load_covariance_matrix(config)
        assert isinstance(result, np.ndarray)

    def test_values_match(self, tmp_path):
        csv_file = tmp_path / "cov.csv"
        np.savetxt(csv_file, REFERENCE_MATRIX, delimiter=",")
        config = {"cov_matrix_path": str(csv_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_shape_is_correct(self, tmp_path):
        csv_file = tmp_path / "cov.csv"
        np.savetxt(csv_file, REFERENCE_MATRIX, delimiter=",")
        config = {"cov_matrix_path": str(csv_file)}
        result = load_covariance_matrix(config)
        assert result.shape == REFERENCE_MATRIX.shape

    def test_unknown_extension_treated_as_csv(self, tmp_path):
        """An unrecognised extension should fall through to the CSV loader."""
        txt_file = tmp_path / "cov.dat"
        np.savetxt(txt_file, REFERENCE_MATRIX, delimiter=",")
        config = {"cov_matrix_path": str(txt_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result, REFERENCE_MATRIX)

    def test_logs_info_when_loading_from_csv(self, tmp_path, caplog):
        csv_file = tmp_path / "cov.csv"
        np.savetxt(csv_file, REFERENCE_MATRIX, delimiter=",")
        config = {"cov_matrix_path": str(csv_file)}
        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config)
        assert any(str(csv_file) in msg for msg in caplog.messages)


# ===========================================================================
# Returns None + warning when neither key is found
# ===========================================================================


class TestReturnsNoneWhenKeysMissing:

    def test_returns_none_for_empty_config(self):
        result = load_covariance_matrix({})
        assert result is None

    def test_returns_none_when_key_matrix_is_none_and_no_path(self):
        result = load_covariance_matrix({}, key_matrix=None)
        assert result is None

    def test_returns_none_when_unrelated_keys_in_config(self):
        config = {"something_else": "value", "another_key": 42}
        result = load_covariance_matrix(config, key_matrix="my_cov")
        assert result is None

    def test_logs_warning_when_neither_key_found(self, caplog):
        config = {}
        with caplog.at_level(logging.WARNING):
            load_covariance_matrix(config, key_matrix="my_cov")
        assert any(
            "my_cov" in msg or "cov_matrix_path" in msg for msg in caplog.messages
        )

    def test_warning_mentions_both_missing_keys(self, caplog):
        config = {}
        with caplog.at_level(logging.WARNING):
            load_covariance_matrix(config, key_matrix="my_cov", key_path="my_path")
        warning_text = " ".join(caplog.messages)
        assert "my_cov" in warning_text or "my_path" in warning_text

    def test_returns_none_when_key_path_value_is_falsy(self, tmp_path):
        """key_path key present but value is empty string — should return None."""
        config = {"cov_matrix_path": ""}
        result = load_covariance_matrix(config)
        assert result is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_1x1_matrix_from_npy(self, tmp_path):
        mat = np.array([[3.14]])
        npy_file = tmp_path / "tiny.npy"
        np.save(npy_file, mat)
        config = {"cov_matrix_path": str(npy_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result, mat)

    def test_large_matrix_from_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 50))
        mat = A @ A.T  # SPD matrix
        csv_file = tmp_path / "large.csv"
        np.savetxt(csv_file, mat, delimiter=",")
        config = {"cov_matrix_path": str(csv_file)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(result, mat)

    def test_key_matrix_none_does_not_match_config_key_none(self):
        """key_matrix=None must NOT accidentally match a config key of None."""
        # dict lookup `None in config` where config has only string keys → False
        config = {"cov_matrix_path": None}
        result = load_covariance_matrix(config, key_matrix=None)
        assert result is None

    @pytest.mark.parametrize(
        "ext,save_fn",
        [
            ("npy", lambda p, m: np.save(p, m)),
            ("csv", lambda p, m: np.savetxt(p, m, delimiter=",")),
        ],
    )
    def test_parametrized_file_formats(self, tmp_path, ext, save_fn):
        mat = np.eye(4) * 7.0
        file_path = tmp_path / f"cov.{ext}"
        save_fn(file_path, mat)
        config = {"cov_matrix_path": str(file_path)}
        result = load_covariance_matrix(config)
        np.testing.assert_array_almost_equal(np.asarray(result), mat)
