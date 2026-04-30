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
Tests for the load_covariance_matrix utility function (new API).

Contract (current implementation):
  - config MUST contain `key` (default: 'cov_matrix') else KeyError
  - If config[key] is None: return None (and log INFO)
  - If config[key] is a path (str/Path): load from file
      * .npy -> np.load
      * .tsv -> pd.read_csv(...).to_numpy()
      * else -> np.loadtxt(..., delimiter=",")
  - If config[key] is array-like (list/tuple/np.ndarray): return np.ndarray
  - Else: ValueError
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sysvar.utils import load_covariance_matrix


REFERENCE_MATRIX = np.array(
    [
        [4.0, 1.0, 0.5],
        [1.0, 9.0, 2.0],
        [0.5, 2.0, 16.0],
    ]
)


class TestKeyHandling:

    def test_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="cov_matrix"):
            load_covariance_matrix({})

    def test_custom_key_missing_raises_keyerror(self):
        with pytest.raises(KeyError, match="my_key"):
            load_covariance_matrix({}, key="my_key")

    def test_none_returns_none_and_logs_info(self, caplog):
        config = {"cov_matrix": None}
        with caplog.at_level(logging.INFO):
            result = load_covariance_matrix(config)
        assert result is None
        assert any(
            "No covariance matrix" in msg or "is None" in msg for msg in caplog.messages
        )


class TestLoadFromArrayLike:

    def test_list_returns_ndarray(self):
        config = {"cov_matrix": REFERENCE_MATRIX.tolist()}
        result = load_covariance_matrix(config)
        assert isinstance(result, np.ndarray)

    def test_list_values_match(self):
        config = {"cov_matrix": REFERENCE_MATRIX.tolist()}
        result = load_covariance_matrix(config)
        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_ndarray_values_match(self):
        config = {"cov_matrix": REFERENCE_MATRIX}
        result = load_covariance_matrix(config)
        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_logs_info_when_loading_from_arraylike(self, caplog):
        config = {"cov_matrix": REFERENCE_MATRIX.tolist()}
        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config)
        assert any("config value" in msg for msg in caplog.messages)


class TestLoadFromPaths:

    def test_path_not_found_raises_valueerror(self, tmp_path):
        config = {"cov_matrix": str(tmp_path / "nope.npy")}
        with pytest.raises(ValueError, match="file not found"):
            load_covariance_matrix(config)

    def test_loads_from_npy(self, tmp_path):
        p = tmp_path / "cov.npy"
        np.save(p, REFERENCE_MATRIX)

        config = {"cov_matrix": str(p)}
        result = load_covariance_matrix(config)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_loads_from_npy_path_object(self, tmp_path):
        p = tmp_path / "cov.npy"
        np.save(p, REFERENCE_MATRIX)

        config = {"cov_matrix": p}  # Path object
        result = load_covariance_matrix(config)

        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_loads_from_tsv(self, tmp_path):
        p = tmp_path / "cov.tsv"
        pd.DataFrame(REFERENCE_MATRIX).to_csv(p, sep="\t", index=False, header=False)

        config = {"cov_matrix": str(p)}
        result = load_covariance_matrix(config)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_unknown_suffix_treated_as_csv_loadtxt(self, tmp_path):
        p = tmp_path / "cov.dat"
        np.savetxt(p, REFERENCE_MATRIX, delimiter=",")

        config = {"cov_matrix": str(p)}
        result = load_covariance_matrix(config)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, REFERENCE_MATRIX)

    def test_logs_info_when_loading_from_file(self, tmp_path, caplog):
        p = tmp_path / "cov.npy"
        np.save(p, REFERENCE_MATRIX)
        config = {"cov_matrix": str(p)}

        with caplog.at_level(logging.INFO):
            load_covariance_matrix(config)

        assert any(str(p) in msg for msg in caplog.messages)


class TestInvalidTypes:

    @pytest.mark.parametrize("bad", [123, 12.3, object(), {"a": 1}])
    def test_invalid_type_raises_valueerror(self, bad):
        config = {"cov_matrix": bad}
        with pytest.raises(ValueError, match="Unsupported type"):
            load_covariance_matrix(config)
