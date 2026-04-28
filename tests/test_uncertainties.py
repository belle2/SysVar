from __future__ import annotations

import pytest
import numpy as np

from sysvar import uncertainties
from sysvar.uncertainties import (
    ExplicitlyCorrelatedUncertainty,
    FullyCorrelatedUncertaintyInParts,
)


shared_params = [
    ("test_valid", np.array([0.2, 0.5, 0.3]), ["[ 0 - 2 ]", "[ 1 - 3 ]", "[ 2 - 4 ]"]),
]

subclass_params = {
    "fully_correlated_in_parts": {"part_dimensions": [1, 2]},
    "explicitly_correlated": {
        "explicit_cov_matrix": np.array(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
        )
    },
}


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainties.get_uncertainty_types().keys()
        for name, errors, string_boundaries in shared_params
    ],
)
def test_uncertainty_initialization(uncertainty_type, name, errors, string_boundaries):
    """Test initialization of uncertainty instances.

    Args:
        uncertainty_type (str): The type of uncertainty (e.g., 'uncorrelated').
        name (str): Name of the uncertainty instance.
        errors (np.ndarray): Array of error values.
        string_boundaries (List[str]): List of string representations of intervals.

    Asserts:
        The instance is correctly initialized with expected name and error array.
    """
    uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
    extra_params = subclass_params.get(uncertainty_type, {})

    if uncertainty_type != "fully_correlated_in_parts":
        uncertainty = uncertainty_class(
            name=name,
            errors=errors,
            string_boundaries=string_boundaries,
            **extra_params,
        )
        assert uncertainty.name == name
        np.testing.assert_array_equal(uncertainty.errors, errors)
    else:
        with pytest.raises(NotImplementedError):
            uncertainty = uncertainty_class(
                name=name,
                errors=errors,
                string_boundaries=string_boundaries,
                **extra_params,
            )


def test_uncertainty_initialization_fully_correlated_in_parts_raise_not_implemented():
    """Test initialization of uncertainty instances.

    Args:
        uncertainty_type (str): The type of uncertainty (e.g., 'uncorrelated').
        name (str): Name of the uncertainty instance.
        errors (np.ndarray): Array of error values.
        string_boundaries (List[str]): List of string representations of intervals.

    Asserts:
        The instance is correctly initialized with expected name and error array.
    """
    uncertainty_class = FullyCorrelatedUncertaintyInParts
    extra_params = subclass_params.get("fully_correlated_in_parts", {})
    name, errors, string_boundaries = shared_params[0]
    with pytest.raises(NotImplementedError):
        uncertainty = uncertainty_class(
            name=name,
            errors=errors,
            string_boundaries=string_boundaries,
            **extra_params,
        )


shared_params_invalid_errors = [
    ("test_tuple_array", (0.1, 0.5, 0.3), ["[ 0 - 2 ]", "[ 1 - 3 ]", "[ 2 - 4 ]"]),
    ("test_string_array", "0.1, 0.5,0.3", ["[ 0 - 2 ]", "[ 1 - 3 ]", "[ 2 - 4 ]"]),
    (
        "test_dict_array",
        {"error_1": 0.1, "error_2": 0.5, "error_3": 0.3},
        ["[ 0 - 2 ]", "[ 1 - 3 ]", "[ 2 - 4 ]"],
    ),
]


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainties.get_uncertainty_types().keys()
        for name, errors, string_boundaries in shared_params_invalid_errors
    ],
)
def test_invalid_errors_type_raises(uncertainty_type, name, errors, string_boundaries):
    """Test that non-array inputs for errors raise NotAnArrayError.

    Args:
        uncertainty_type (str): Type of uncertainty.
        name (str): Name of the uncertainty instance.
        errors (Any): Invalid errors input.
        string_boundaries (List[str]): List of string interval representations.

    Raises:
        NotAnArrayError: If `errors` is not a NumPy array.
    """
    with pytest.raises(
        uncertainties.NotAnArrayError, match="The errors must be provided as np arrays"
    ):
        uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
        extra_params = subclass_params.get(uncertainty_type, {})
        _ = uncertainty_class(
            name=name,
            errors=errors,
            string_boundaries=string_boundaries,
            **extra_params,
        )


shared_params_invalid_shape = [
    (
        "test_invalid_shape",
        np.array([[0.1, 0.5], [0.3, 0.2]]),
        ["[ 0 - 2 ]", "[ 1 - 3 ]"],
    ),
]


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainties.get_uncertainty_types().keys()
        for name, errors, string_boundaries in shared_params_invalid_shape
    ],
)
def test_multidimensional_errors_raise(
    uncertainty_type, name, errors, string_boundaries
):
    """Test that multidimensional arrays raise MultiDimArrayError.

    Args:
        uncertainty_type (str): The type of uncertainty.
        name (str): Name of the instance.
        errors (np.ndarray): Multidimensional array of errors.
        string_boundaries (List[str]): Interval strings.

    Raises:
        MultiDimArrayError: If `errors` is not a 1D array.
    """
    with pytest.raises(
        uncertainties.MultiDimArrayError,
        match="The errors must be provided as a 1D array",
    ):
        uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
        extra_params = subclass_params.get(uncertainty_type, {})
        _ = uncertainty_class(
            name=name,
            errors=errors,
            string_boundaries=string_boundaries,
            **extra_params,
        )


shared_params_empty_array = [
    ("test_empty_array", np.array([]), ["[ 0 - 2 ]", "[ 1 - 3 ]"]),
]


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainties.get_uncertainty_types().keys()
        for name, errors, string_boundaries in shared_params_empty_array
    ],
)
def test_empty_array_raises(uncertainty_type, name, errors, string_boundaries):
    """Test that an empty array for errors raises EmptyArrayError.

    Args:
        uncertainty_type (str): Type of uncertainty.
        name (str): Instance name.
        errors (np.ndarray): Empty array.
        string_boundaries (List[str]): Interval strings.

    Raises:
        EmptyArrayError: If `errors` is empty.
    """
    with pytest.raises(
        uncertainties.EmptyArrayError, match="The errors array cannot be empty"
    ):
        uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
        extra_params = subclass_params.get(uncertainty_type, {})
        _ = uncertainty_class(
            name=name,
            errors=errors,
            string_boundaries=string_boundaries,
            **extra_params,
        )


shared_params_covmatrix_dim = [
    ("test_covmatrix_dim", np.array([0.5, 0.2, 0.3]), ["[0-1]", "[1-2]", "[2-3]"]),
]


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainties.get_uncertainty_types().keys()
        for name, errors, string_boundaries in shared_params_covmatrix_dim
    ],
)
def test_covariance_shape_and_type(uncertainty_type, name, errors, string_boundaries):
    """Test that covariance matrix is a square NumPy array of expected shape.

    Args:
        uncertainty_type (str): Type of uncertainty.
        name (str): Name of the instance.
        errors (np.ndarray): Errors array.
        string_boundaries (List[str]): Interval strings.

    Asserts:
        Covariance matrix is a NumPy array with shape (n, n), where n is length of `errors`.
    """
    uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
    extra_params = subclass_params.get(uncertainty_type, {})
    uncertainty = uncertainty_class(
        name=name, errors=errors, string_boundaries=string_boundaries, **extra_params
    )

    cov = uncertainty.cov_matrix
    assert isinstance(cov, np.ndarray)
    assert cov.shape == (len(errors), len(errors))


uncertainty_types = ["fully_correlated", "uncorrelated"]

shared_params_corrmatrix = [
    ("test_corrmatrix", np.array([0.2, 0.3, 0.3]), ["[0-1]", "[1-2]", "[2-3]"]),
]


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, name, errors, string_boundaries)
        for ut in uncertainty_types
        for name, errors, string_boundaries in shared_params_corrmatrix
    ],
)
def test_correlation_matrix_values(uncertainty_type, name, errors, string_boundaries):
    """Test that correlation matrix is correctly set for specific uncertainty types.

    Args:
        uncertainty_type (str): One of 'fully_correlated' or 'uncorrelated'.
        name (str): Instance name.
        errors (np.ndarray): Errors array.
        string_boundaries (List[str]): Interval strings.

    Asserts:
        Correlation matrix is identity for 'uncorrelated' and all ones for 'fully_correlated'.
    """
    uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
    uncertainty = uncertainty_class(
        name=name, errors=errors, string_boundaries=string_boundaries
    )

    if uncertainty_type == "fully_correlated":
        expected = np.ones((3, 3))
    elif uncertainty_type == "uncorrelated":
        expected = np.eye(3)

    np.testing.assert_array_equal(uncertainty.corr_matrix, expected)


@pytest.mark.parametrize(
    "name, errors, string_boundaries, cov_matrix",
    [
        (
            "explicit_cov_test",
            np.array([1.0, 2.0, 3.0]),
            ["[0-1]", "[1-2]", "[2-3]"],
            np.array([[1.0, 0.5, 0.3], [0.5, 4.0, 0.6], [0.3, 0.6, 9.0]]),
        )
    ],
)
def test_explicitly_correlated_correct_covariance(
    name, errors, string_boundaries, cov_matrix
):
    """Test explicitly correlated uncertainty retains the given covariance matrix.

    Args:
        name (str): Instance name.
        errors (np.ndarray): Errors array.
        string_boundaries (List[str]): Interval strings.
        cov_matrix (np.ndarray): Provided covariance matrix.

    Asserts:
        The instance's covariance matrix matches the given explicit matrix.
    """

    uncertainty = ExplicitlyCorrelatedUncertainty(
        name=name,
        errors=errors,
        string_boundaries=string_boundaries,
        explicit_cov_matrix=cov_matrix,
    )
    np.testing.assert_array_equal(uncertainty.cov_matrix, cov_matrix)


@pytest.mark.parametrize(
    "uncertainty_type, errors, expected_cov, extra",
    [
        (
            "uncorrelated",
            np.array([1.0, 2.0, 3.0]),
            np.diag([1.0**2, 2.0**2, 3.0**2]),
            {},
        ),
        (
            "fully_correlated",
            np.array([1.0, 2.0, 3.0]),
            np.diag([1.0, 2.0, 3.0]) @ np.ones((3, 3)) @ np.diag([1.0, 2.0, 3.0]),
            {},
        ),
        (
            "explicitly_correlated",
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 0.5, 0.3], [0.5, 4.0, 0.6], [0.3, 0.6, 9.0]]),
            {
                "explicit_cov_matrix": np.array(
                    [[1.0, 0.5, 0.3], [0.5, 4.0, 0.6], [0.3, 0.6, 9.0]]
                )
            },
        ),
        (
            "fully_correlated_in_parts",
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 6.0], [0.0, 6.0, 9.0]]),
            {"part_dimensions": [1, 2]},
        ),
    ],
)
def test_expected_covariance_per_subclass(
    uncertainty_type, errors, expected_cov, extra
):
    """Test expected covariance matrix per subclass implementation.

    Args:
        uncertainty_type (str): Type of uncertainty.
        errors (np.ndarray): Errors array.
        expected_cov (np.ndarray): Expected covariance matrix.
        extra (dict): Extra parameters needed by the subclass.

    Asserts:
        Actual covariance matrix matches the expected one.
    """
    string_boundaries = ["[0-1]", "[1-2]", "[2-3]"]
    if uncertainty_type != "fully_correlated_in_parts":
        uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]

        uncertainty = uncertainty_class(
            name="test", errors=errors, string_boundaries=string_boundaries, **extra
        )

        np.testing.assert_allclose(uncertainty.cov_matrix, expected_cov)
    else:
        uncertainty_class = FullyCorrelatedUncertaintyInParts
        with pytest.raises(NotImplementedError):
            uncertainty = uncertainty_class(
                name="test", errors=errors, string_boundaries=string_boundaries, **extra
            )
    string_boundaries = ["[0-1]", "[1-2]", "[2-3]"]


@pytest.mark.parametrize(
    ("uncertainty_type", "bad_value"),
    [
        (ut, bad_value)
        for ut in uncertainties.get_uncertainty_types().keys()
        for bad_value in [np.nan, np.inf, -np.inf]
    ],
)
def test_invalid_error_values(uncertainty_type, bad_value):
    """Test that invalid numeric values (NaN, inf) raise ValueError.

    Args:
        uncertainty_type (str): Type of uncertainty.
        bad_value (float): Invalid error value to test.

    Raises:
        ValueError: If `errors` contains NaN or infinite values.
    """
    with pytest.raises(
        uncertainties.ValueError, match="The errors array contains invalid values"
    ):
        uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
        bad_errors = np.array([bad_value, 1.0, 2.0])
        extra_params = subclass_params.get(uncertainty_type, {})
        _ = uncertainty_class(
            name="bad_uncertainty",
            errors=bad_errors,
            string_boundaries=["[0-1]", "[1-2]", "[2-3]"],
            **extra_params,
        )


@pytest.mark.parametrize(
    ("uncertainty_type", "name", "errors", "string_boundaries"),
    [
        (ut, "symmetry_test", np.array([0.5, 0.2, 0.3]), ["[0-1]", "[1-2]", "[2-3]"])
        for ut in uncertainties.get_uncertainty_types().keys()
    ],
)
def test_covariance_matrix_is_symmetric(
    uncertainty_type, name, errors, string_boundaries
):
    """Test that covariance matrix is symmetric.

    Args:
        uncertainty_type (str): Type of uncertainty.
        name (str): Name of the uncertainty instance.
        errors (np.ndarray): Errors array.
        string_boundaries (List[str]): Interval strings.

    Asserts:
        The covariance matrix is square and symmetric.
    """
    uncertainty_class = uncertainties.get_uncertainty_types()[uncertainty_type]
    extra_params = subclass_params.get(uncertainty_type, {})
    uncertainty = uncertainty_class(
        name=name, errors=errors, string_boundaries=string_boundaries, **extra_params
    )

    cov = uncertainty.cov_matrix

    # Check that the covariance matrix is symmetric
    assert cov.shape[0] == cov.shape[1], "Covariance matrix must be square"
    np.testing.assert_array_almost_equal(
        cov, cov.T, err_msg=f"{uncertainty_type} covariance matrix is not symmetric"
    )
