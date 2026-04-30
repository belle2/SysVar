from pathlib import Path

import pytest
import sysvar.corrections as corrections


def _repo_root_from_test_file() -> Path:
    # tests/ is at <repo_root>/tests
    return Path(__file__).resolve().parents[1]


def _config_csv_path(csv_filename: str) -> Path:
    return _repo_root_from_test_file() / "configs" / "csv_configs" / csv_filename


# -----------------------------
# 1D CSV structure smoke tests
# -----------------------------


@pytest.fixture(
    params=[
        {"csv": "charged_slow_pi_correction.csv", "cov": None},
        {"csv": "fei_Bp_001.csv", "cov": "Comb_Cov_RD_Bp_SigProb_0_001.npy"},
    ]
)
def corr1d_and_path(request):
    csv_filename = request.param["csv"]
    cov_filename = request.param["cov"]

    csv_path = _config_csv_path(csv_filename)
    if not csv_path.exists():
        pytest.skip(f"CSV config not found in repo: {csv_path}")

    cov_path = None
    if cov_filename is not None:
        cov_path = _config_csv_path(cov_filename)
        if not cov_path.exists():
            pytest.skip(f"Cov matrix config not found in repo: {cov_path}")

    corr = corrections.Correction1DFromCSV(
        csv_path=str(csv_path),
        cov_matrix_path=(None if cov_path is None else str(cov_path)),
    )
    return corr, csv_path


def test_1d_csv_loads_table(corr1d_and_path):
    corr, csv_path = corr1d_and_path
    assert corr.table is not None, f"table is None for {csv_path}"
    assert len(corr.table) > 0, f"table is empty for {csv_path}"


def test_1d_csv_has_dependant_variable(corr1d_and_path):
    corr, csv_path = corr1d_and_path
    assert isinstance(
        corr.dependant_variable, str
    ), f"dependant_variable type invalid for {csv_path}"
    assert (
        corr.dependant_variable.strip() != ""
    ), f"dependant_variable blank for {csv_path}"


def test_1d_csv_central_values_length_matches_rows(corr1d_and_path):
    corr, csv_path = corr1d_and_path
    assert corr.central_values is not None, f"central_values is None for {csv_path}"
    assert len(corr.central_values) == len(corr.table), (
        f"central_values length != table rows for {csv_path}: "
        f"{len(corr.central_values)} vs {len(corr.table)}"
    )


def test_1d_csv_build_queries_returns_one_per_row(corr1d_and_path):
    corr, csv_path = corr1d_and_path
    queries = corr.build_queries(prefix=None)

    assert isinstance(
        queries, list
    ), f"build_queries did not return list for {csv_path}"
    assert len(queries) == len(
        corr.table
    ), f"queries length != table rows for {csv_path}: {len(queries)} vs {len(corr.table)}"
    assert all(
        isinstance(q, str) and q.strip() != "" for q in queries
    ), f"one or more queries are empty/non-string for {csv_path}: {queries}"


def test_1d_csv_mode_is_consistent(corr1d_and_path):
    corr, csv_path = corr1d_and_path

    if corr.use_equality_queries:
        assert (
            corr.variable_values is not None
        ), f"variable_values is None in equality mode for {csv_path}"
        assert len(corr.variable_values) == len(corr.table), (
            f"variable_values length != table rows for {csv_path}: "
            f"{len(corr.variable_values)} vs {len(corr.table)}"
        )
    else:
        assert (
            corr.lower_bounds is not None
        ), f"lower_bounds is None in binned mode for {csv_path}"
        assert (
            corr.upper_bounds is not None
        ), f"upper_bounds is None in binned mode for {csv_path}"
        assert len(corr.lower_bounds) == len(corr.table), (
            f"lower_bounds length != table rows for {csv_path}: "
            f"{len(corr.lower_bounds)} vs {len(corr.table)}"
        )
        assert len(corr.upper_bounds) == len(corr.table), (
            f"upper_bounds length != table rows for {csv_path}: "
            f"{len(corr.upper_bounds)} vs {len(corr.table)}"
        )


def test_1d_csv_populates_uncertainties(corr1d_and_path):
    corr, csv_path = corr1d_and_path
    assert isinstance(
        corr.uncertainties, dict
    ), f"uncertainties is not a dict for {csv_path}"
    assert (
        len(corr.uncertainties) >= 1
    ), f"no uncertainties were populated for {csv_path}"


# -----------------------------
# 2D CSV structure smoke tests
# -----------------------------


@pytest.fixture(
    params=[
        # neutral pi 2D: no covariance matrix (adjust/add entries as needed)
        {"csv": "neutral_pi_corr.csv", "cov": None},
        # Example if you later add a 2D cov matrix:
        # {"csv": "some_2d.csv", "cov": "some_2d_cov.npy"},
    ]
)
def corr2d_and_path(request):
    """
    Parametrized fixture for 2D CSVs, with optional covariance matrix file.
    """
    csv_filename = request.param["csv"]
    cov_filename = request.param["cov"]

    csv_path = _config_csv_path(csv_filename)
    if not csv_path.exists():
        pytest.skip(f"CSV config not found in repo: {csv_path}")

    cov_path = None
    if cov_filename is not None:
        cov_path = _config_csv_path(cov_filename)
        if not cov_path.exists():
            pytest.skip(f"Cov matrix config not found in repo: {cov_path}")

    corr = corrections.Correction2DFromCSV(
        csv_path=str(csv_path),
        cov_matrix_path=(None if cov_path is None else str(cov_path)),
    )
    return corr, csv_path


def test_2d_csv_loads_table(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    assert corr.table is not None, f"table is None for {csv_path}"
    assert len(corr.table) > 0, f"table is empty for {csv_path}"


def test_2d_csv_has_both_dependant_variables(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    assert isinstance(
        corr.dependant_variable_1, str
    ), f"dependant_variable_1 type invalid for {csv_path}"
    assert isinstance(
        corr.dependant_variable_2, str
    ), f"dependant_variable_2 type invalid for {csv_path}"
    assert (
        corr.dependant_variable_1.strip() != ""
    ), f"dependant_variable_1 blank for {csv_path}"
    assert (
        corr.dependant_variable_2.strip() != ""
    ), f"dependant_variable_2 blank for {csv_path}"


def test_2d_csv_units_are_strings(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    assert isinstance(corr.unit_1, str), f"unit_1 is not a string for {csv_path}"
    assert isinstance(corr.unit_2, str), f"unit_2 is not a string for {csv_path}"


def test_2d_csv_edges_columns_exist_and_iterator_matches_rows(corr2d_and_path):
    corr, csv_path = corr2d_and_path

    for col in (corr._v1_min, corr._v1_max, corr._v2_min, corr._v2_max):
        assert (
            col in corr.table.columns
        ), f"missing expected bin-edge column '{col}' in {csv_path}"

    bins = list(corr.iterator)
    assert len(bins) == len(corr.table), f"iterator length != table rows for {csv_path}"
    assert all(
        len(b) == 4 for b in bins
    ), f"iterator did not yield 4-tuples for {csv_path}"


def test_2d_csv_central_values_length_matches_rows(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    assert corr.central_values is not None, f"central_values is None for {csv_path}"
    assert len(corr.central_values) == len(corr.table), (
        f"central_values length != table rows for {csv_path}: "
        f"{len(corr.central_values)} vs {len(corr.table)}"
    )


def test_2d_csv_build_queries_returns_one_per_row(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    queries = corr.build_queries(prefix=None)

    assert isinstance(
        queries, list
    ), f"build_queries did not return list for {csv_path}"
    assert len(queries) == len(
        corr.table
    ), f"queries length != table rows for {csv_path}: {len(queries)} vs {len(corr.table)}"
    assert all(
        isinstance(q, str) and q.strip() != "" for q in queries
    ), f"one or more queries are empty/non-string for {csv_path}: {queries}"


def test_2d_csv_visual_labels_returns_one_per_row(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    labels = corr.visual_labels

    assert isinstance(labels, list), f"visual_labels did not return list for {csv_path}"
    assert len(labels) == len(
        corr.table
    ), f"visual_labels length != table rows for {csv_path}: {len(labels)} vs {len(corr.table)}"
    assert all(
        isinstance(l, str) and l.strip() != "" for l in labels
    ), f"one or more visual_labels are empty/non-string for {csv_path}: {labels}"


def test_2d_csv_populates_uncertainties(corr2d_and_path):
    corr, csv_path = corr2d_and_path
    assert isinstance(
        corr.uncertainties, dict
    ), f"uncertainties is not a dict for {csv_path}"
    assert (
        len(corr.uncertainties) >= 1
    ), f"no uncertainties were populated for {csv_path}"


# -----------------------------
# 3D CSV structure smoke tests
# -----------------------------


@pytest.fixture(
    params=[
        # Kshort 3D: no covariance matrix
        {"csv": "Kshort_corr.csv", "cov": None},
    ]
)
def corr3d_and_path(request):
    csv_filename = request.param["csv"]
    cov_filename = request.param["cov"]

    csv_path = _config_csv_path(csv_filename)
    if not csv_path.exists():
        pytest.skip(f"CSV config not found in repo: {csv_path}")

    cov_path = None
    if cov_filename is not None:
        cov_path = _config_csv_path(cov_filename)
        if not cov_path.exists():
            pytest.skip(f"Cov matrix config not found in repo: {cov_path}")

    corr = corrections.Correction3DFromCSV(
        csv_path=str(csv_path),
        cov_matrix_path=(None if cov_path is None else str(cov_path)),
    )
    return corr, csv_path


def test_3d_csv_loads_table(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    assert corr.table is not None, f"table is None for {csv_path}"
    assert len(corr.table) > 0, f"table is empty for {csv_path}"


def test_3d_csv_has_all_dependant_variables(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    assert isinstance(
        corr.dependant_variable_1, str
    ), f"dependant_variable_1 type invalid for {csv_path}"
    assert isinstance(
        corr.dependant_variable_2, str
    ), f"dependant_variable_2 type invalid for {csv_path}"
    assert isinstance(
        corr.dependant_variable_3, str
    ), f"dependant_variable_3 type invalid for {csv_path}"

    assert (
        corr.dependant_variable_1.strip() != ""
    ), f"dependant_variable_1 blank for {csv_path}"
    assert (
        corr.dependant_variable_2.strip() != ""
    ), f"dependant_variable_2 blank for {csv_path}"
    assert (
        corr.dependant_variable_3.strip() != ""
    ), f"dependant_variable_3 blank for {csv_path}"


def test_3d_csv_units_are_strings(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    assert isinstance(corr.unit_1, str), f"unit_1 is not a string for {csv_path}"
    assert isinstance(corr.unit_2, str), f"unit_2 is not a string for {csv_path}"
    assert isinstance(corr.unit_3, str), f"unit_3 is not a string for {csv_path}"


def test_3d_csv_edges_columns_exist_and_iterator_matches_rows(corr3d_and_path):
    corr, csv_path = corr3d_and_path

    for col in (
        corr._v1_min,
        corr._v1_max,
        corr._v2_min,
        corr._v2_max,
        corr._v3_min,
        corr._v3_max,
    ):
        assert (
            col in corr.table.columns
        ), f"missing expected bin-edge column '{col}' in {csv_path}"

    bins = list(corr.iterator)
    assert len(bins) == len(corr.table), f"iterator length != table rows for {csv_path}"
    assert all(
        len(b) == 6 for b in bins
    ), f"iterator did not yield 6-tuples for {csv_path}"


def test_3d_csv_central_values_length_matches_rows(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    assert corr.central_values is not None, f"central_values is None for {csv_path}"
    assert len(corr.central_values) == len(corr.table), (
        f"central_values length != table rows for {csv_path}: "
        f"{len(corr.central_values)} vs {len(corr.table)}"
    )


def test_3d_csv_build_queries_returns_one_per_row(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    queries = corr.build_queries(prefix=None)

    assert isinstance(
        queries, list
    ), f"build_queries did not return list for {csv_path}"
    assert len(queries) == len(
        corr.table
    ), f"queries length != table rows for {csv_path}: {len(queries)} vs {len(corr.table)}"
    assert all(
        isinstance(q, str) and q.strip() != "" for q in queries
    ), f"one or more queries are empty/non-string for {csv_path}: {queries}"


def test_3d_csv_visual_labels_returns_one_per_row(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    labels = corr.visual_labels

    assert isinstance(labels, list), f"visual_labels did not return list for {csv_path}"
    assert len(labels) == len(
        corr.table
    ), f"visual_labels length != table rows for {csv_path}: {len(labels)} vs {len(corr.table)}"
    assert all(
        isinstance(l, str) and l.strip() != "" for l in labels
    ), f"one or more visual_labels are empty/non-string for {csv_path}: {labels}"


def test_3d_csv_populates_uncertainties(corr3d_and_path):
    corr, csv_path = corr3d_and_path
    assert isinstance(
        corr.uncertainties, dict
    ), f"uncertainties is not a dict for {csv_path}"
    assert (
        len(corr.uncertainties) >= 1
    ), f"no uncertainties were populated for {csv_path}"
