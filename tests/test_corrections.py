"""
Tests for the *methods* of the Correction classes (not just input structure).

All tests are self-contained: CSVs and covariance matrices are built in-memory
using pytest's tmp_path fixture, so no real config files are needed.

Coverage:
  - BaseCorrection._build_column_name
  - BaseCorrection.N, total_error, add_uncertainty, populate_uncertainties
  - BaseCorrectionFromCSV._is_valid_table, _build_info_from_table, add_extra_cuts
  - Correction1DFromCSV: build_queries (binned & equality), visual_labels, value_edges/mids,
    get_unit, populate_uncertainties (fully-corr, uncorr, explicit cov)
  - Correction2DFromCSV: iterator, visual_labels, build_queries, add_extra_cuts
  - Correction3DFromCSV: iterator, visual_labels, build_queries
  - Error / exception paths for all classes
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import sysvar.corrections as corrections
from sysvar.corrections import (
    BaseCorrection,
    BaseCorrectionFromCSV,
    Correction1DFromCSV,
    Correction2DFromCSV,
    Correction3DFromCSV,
    MissingInformationError,
    InvalidCorrectionTableKey,
    UncertaintyWithSameNameExists,
    CorrectionBF,
    CustomCorrection,
)


# ===========================================================================
# Helpers – build minimal CSV files on disk
# ===========================================================================


def _write_csv(tmp_path: Path, filename: str, content: str) -> Path:
    """Strip leading whitespace and write content to a temp CSV file."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip())
    return p


def _write_npy(tmp_path: Path, filename: str, matrix: np.ndarray) -> Path:
    p = tmp_path / filename
    np.save(p, matrix)
    return p


# ---------------------------------------------------------------------------
# Minimal 1D binned CSV (no explicit covariance)
# ---------------------------------------------------------------------------
BINNED_1D_CSV = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008
p,1.0,2.0,0.97,GeV,0.01,0.02,0.005,0.008
p,2.0,3.0,1.00,GeV,0.01,0.02,0.005,0.008
"""

# 1D equality (discrete-value) CSV
EQUALITY_1D_CSV = """\
dependant_variable,p,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,0,0.95,GeV,0.01,0.02,0.005,0.008
p,1,0.97,GeV,0.01,0.02,0.005,0.008
p,2,1.00,GeV,0.01,0.02,0.005,0.008
"""

# 1D CSV with PDG extra cuts
PDG_1D_CSV = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,PDG
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,"[521,-521]"
p,1.0,2.0,0.97,GeV,0.01,0.02,0.005,0.008,"[521,-521]"
"""

# 1D CSV with custom label column
LABEL_1D_CSV = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,p_label
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,low
p,1.0,2.0,0.97,GeV,0.01,0.02,0.005,0.008,high
"""

# Minimal 2D CSV
BINNED_2D_CSV = """\
dependant_variable_1,dependant_variable_2,p_min,p_max,cosTheta_min,cosTheta_max,central_value,p_unit,cosTheta_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,cosTheta,0.0,1.0,-1.0,0.0,0.90,GeV,,0.01,0.02,0.005,0.008
p,cosTheta,0.0,1.0,0.0,1.0,0.92,GeV,,0.01,0.02,0.005,0.008
p,cosTheta,1.0,2.0,-1.0,0.0,0.95,GeV,,0.01,0.02,0.005,0.008
p,cosTheta,1.0,2.0,0.0,1.0,0.98,GeV,,0.01,0.02,0.005,0.008
"""

# 2D with PDG extra cuts
PDG_2D_CSV = """\
dependant_variable_1,dependant_variable_2,p_min,p_max,cosTheta_min,cosTheta_max,central_value,p_unit,cosTheta_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,PDG
p,cosTheta,0.0,1.0,-1.0,0.0,0.90,GeV,,0.01,0.02,0.005,0.008,"[521]"
p,cosTheta,0.0,1.0,0.0,1.0,0.92,GeV,,0.01,0.02,0.005,0.008,"[521]"
"""

# Minimal 3D CSV
BINNED_3D_CSV = """\
dependant_variable_1,dependant_variable_2,dependant_variable_3,p_min,p_max,cosTheta_min,cosTheta_max,phi_min,phi_max,central_value,p_unit,cosTheta_unit,phi_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,cosTheta,phi,0.0,1.0,-1.0,0.0,0.0,1.0,0.90,GeV,,,0.01,0.02,0.005,0.008
p,cosTheta,phi,0.0,1.0,0.0,1.0,0.0,1.0,0.92,GeV,,,0.01,0.02,0.005,0.008
"""


# ===========================================================================
# BaseCorrection._build_column_name  (static method, no CSV needed)
# ===========================================================================


class TestBuildColumnName:

    def test_prefix_and_variable(self):
        result = BaseCorrection._build_column_name("slow_pi", "p")
        assert result == "slow_pi_p"

    def test_no_prefix_returns_variable(self):
        result = BaseCorrection._build_column_name(None, "cosTheta")
        assert result == "cosTheta"

    def test_empty_prefix_joins_with_underscore(self):
        # An empty string prefix is still a valid str, joined with "_"
        result = BaseCorrection._build_column_name("", "p")
        assert result == "_p"

    def test_invalid_variable_type_raises(self):
        with pytest.raises(ValueError):
            BaseCorrection._build_column_name("prefix", 123)

    def test_invalid_prefix_type_raises(self):
        with pytest.raises(ValueError):
            BaseCorrection._build_column_name(42, "p")


# ===========================================================================
# Correction1DFromCSV  – construction and properties
# ===========================================================================


@pytest.fixture
def corr1d_binned(tmp_path):
    csv = _write_csv(tmp_path, "binned.csv", BINNED_1D_CSV)
    return Correction1DFromCSV(csv_path=str(csv))


@pytest.fixture
def corr1d_equality(tmp_path):
    csv = _write_csv(tmp_path, "equality.csv", EQUALITY_1D_CSV)
    return Correction1DFromCSV(csv_path=str(csv))


@pytest.fixture
def corr1d_with_cov(tmp_path):
    csv = _write_csv(tmp_path, "binned.csv", BINNED_1D_CSV)
    cov = np.diag([0.01**2, 0.01**2, 0.01**2])
    npy = _write_npy(tmp_path, "cov.npy", cov)
    return Correction1DFromCSV(csv_path=str(csv), cov_matrix_path=str(npy))


@pytest.fixture
def corr1d_pdg(tmp_path):
    csv = _write_csv(tmp_path, "pdg.csv", PDG_1D_CSV)
    return Correction1DFromCSV(csv_path=str(csv))


@pytest.fixture
def corr1d_label(tmp_path):
    csv = _write_csv(tmp_path, "label.csv", LABEL_1D_CSV)
    return Correction1DFromCSV(csv_path=str(csv))


class TestCorrection1DFromCSV_Properties:

    def test_N_equals_number_of_rows(self, corr1d_binned):
        assert corr1d_binned.N == 3

    def test_central_values_are_floats(self, corr1d_binned):
        assert all(isinstance(v, float) for v in corr1d_binned.central_values)

    def test_central_values_correct(self, corr1d_binned):
        np.testing.assert_allclose(corr1d_binned.central_values, [0.95, 0.97, 1.00])

    def test_lower_bounds_correct(self, corr1d_binned):
        assert corr1d_binned.lower_bounds == [0.0, 1.0, 2.0]

    def test_upper_bounds_correct(self, corr1d_binned):
        assert corr1d_binned.upper_bounds == [1.0, 2.0, 3.0]

    def test_unit_is_read_from_csv(self, corr1d_binned):
        assert corr1d_binned.unit == "GeV"

    def test_dependant_variable_is_correct(self, corr1d_binned):
        assert corr1d_binned.dependant_variable == "p"

    def test_use_equality_queries_false_for_binned(self, corr1d_binned):
        assert corr1d_binned.use_equality_queries is False

    def test_use_equality_queries_true_for_discrete(self, corr1d_equality):
        assert corr1d_equality.use_equality_queries is True

    def test_variable_values_set_for_equality_mode(self, corr1d_equality):
        assert corr1d_equality.variable_values == [0, 1, 2]

    def test_value_edges_binned(self, corr1d_binned):
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(corr1d_binned.value_edges, expected)

    def test_value_mids_binned(self, corr1d_binned):
        expected = np.array([0.5, 1.5, 2.5])
        np.testing.assert_array_almost_equal(corr1d_binned.value_mids, expected)

    def test_value_edges_equality(self, corr1d_equality):
        # equality mode returns arange(n+1)
        np.testing.assert_array_equal(
            corr1d_equality.value_edges, np.array([0, 1, 2, 3])
        )

    def test_value_mids_equality(self, corr1d_equality):
        np.testing.assert_array_equal(corr1d_equality.value_mids, np.array([0, 1, 2]))


class TestCorrection1DFromCSV_VisualLabels:

    def test_visual_labels_count(self, corr1d_binned):
        assert len(corr1d_binned.visual_labels) == 3

    def test_visual_labels_contain_variable_name(self, corr1d_binned):
        for label in corr1d_binned.visual_labels:
            assert "p" in label

    def test_visual_labels_contain_unit(self, corr1d_binned):
        for label in corr1d_binned.visual_labels:
            assert "GeV" in label

    def test_visual_labels_binned_format(self, corr1d_binned):
        assert corr1d_binned.visual_labels[0] == "0.0 < p < 1.0 GeV"

    def test_visual_labels_equality_format(self, corr1d_equality):
        assert corr1d_equality.visual_labels[0] == "p = 0 GeV"

    def test_visual_labels_from_label_column(self, corr1d_label):
        assert corr1d_label.visual_labels == ["low", "high"]


class TestCorrection1DFromCSV_BuildQueries:

    def test_build_queries_no_prefix_binned(self, corr1d_binned):
        queries = corr1d_binned.build_queries(prefix=None)
        assert queries[0] == "0.0 <= p < 1.0"
        assert queries[1] == "1.0 <= p < 2.0"
        assert queries[2] == "2.0 <= p < 3.0"

    def test_build_queries_with_prefix_binned(self, corr1d_binned):
        queries = corr1d_binned.build_queries(prefix="slow_pi")
        assert all("slow_pi_p" in q for q in queries)

    def test_build_queries_equality_no_prefix(self, corr1d_equality):
        queries = corr1d_equality.build_queries(prefix=None)
        assert queries[0] == "p == 0"
        assert queries[1] == "p == 1"
        assert queries[2] == "p == 2"

    def test_build_queries_equality_with_prefix(self, corr1d_equality):
        queries = corr1d_equality.build_queries(prefix="trk")
        assert all("trk_p ==" in q for q in queries)

    def test_build_queries_returns_list(self, corr1d_binned):
        assert isinstance(corr1d_binned.build_queries(), list)

    def test_build_queries_length_matches_rows(self, corr1d_binned):
        assert len(corr1d_binned.build_queries()) == corr1d_binned.N

    def test_build_queries_all_non_empty_strings(self, corr1d_binned):
        for q in corr1d_binned.build_queries():
            assert isinstance(q, str) and q.strip()


class TestCorrection1DFromCSV_PDGExtraCuts:

    def test_pdg_multi_value_uses_in_operator(self, corr1d_pdg):
        queries = corr1d_pdg.build_queries(prefix=None)
        for q in queries:
            assert "PDG in" in q or "PDG ==" in q

    def test_pdg_query_contains_base_condition(self, corr1d_pdg):
        queries = corr1d_pdg.build_queries(prefix=None)
        # Base condition must still be present
        for q in queries:
            assert "<=" in q or "==" in q

    def test_pdg_with_prefix_uses_prefixed_column(self, corr1d_pdg):
        queries = corr1d_pdg.build_queries(prefix="slow_pi")
        for q in queries:
            assert "slow_pi_PDG" in q

    def test_single_pdg_uses_equality(self, tmp_path):
        csv_content = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,PDG
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,[521]
"""
        csv = _write_csv(tmp_path, "single_pdg.csv", csv_content)
        corr = Correction1DFromCSV(csv_path=str(csv))
        queries = corr.build_queries(prefix=None)
        assert "PDG == 521" in queries[0]

    def test_mcpdg_column_is_also_handled(self, tmp_path):
        csv_content = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,mcPDG
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,[521]
"""
        csv = _write_csv(tmp_path, "mcpdg.csv", csv_content)
        corr = Correction1DFromCSV(csv_path=str(csv))
        queries = corr.build_queries(prefix=None)
        assert "mcPDG ==" in queries[0]

    def test_nan_pdg_row_adds_no_extra_condition(self, tmp_path):
        csv_content = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,PDG
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,
p,1.0,2.0,0.97,GeV,0.01,0.02,0.005,0.008,
"""
        csv = _write_csv(tmp_path, "nan_pdg.csv", csv_content)
        corr = Correction1DFromCSV(csv_path=str(csv))
        queries = corr.build_queries(prefix=None)
        # No PDG condition should be injected for NaN rows
        assert "PDG" not in queries[0]
        assert "PDG" not in queries[1]


class TestCorrection1DFromCSV_Uncertainties:

    def test_uncertainties_populated_from_columns(self, corr1d_binned):
        assert len(corr1d_binned.uncertainties) >= 1

    def test_total_error_length_matches_N(self, corr1d_binned):
        assert len(corr1d_binned.total_error) == corr1d_binned.N

    def test_total_error_is_positive(self, corr1d_binned):
        assert np.all(corr1d_binned.total_error > 0)

    def test_explicit_cov_creates_single_uncertainty(self, corr1d_with_cov):
        assert "explicit_covariance" in corr1d_with_cov.uncertainties

    def test_explicit_cov_total_error_matches_diagonal(self, corr1d_with_cov):
        cov = np.diag([0.01**2, 0.01**2, 0.01**2])
        expected_errors = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(
            corr1d_with_cov.total_error, expected_errors, rtol=1e-6
        )

    def test_add_duplicate_uncertainty_raises(self, corr1d_binned):
        from sysvar.uncertainties import FullyCorrelatedUncertainty

        with pytest.raises(UncertaintyWithSameNameExists):
            existing_name = next(iter(corr1d_binned.uncertainties))
            corr1d_binned.add_uncertainty(
                unc_name=existing_name,
                unc_values=[0.01] * corr1d_binned.N,
                unc_obj=FullyCorrelatedUncertainty,
            )

    def test_no_uncertainties_total_error_raises(self, tmp_path):
        # Build a correction then clear its uncertainties manually
        csv = _write_csv(tmp_path, "binned.csv", BINNED_1D_CSV)
        corr = Correction1DFromCSV(csv_path=str(csv))
        corr.uncertainties = {}
        with pytest.raises(ValueError):
            _ = corr.total_error


# ===========================================================================
# Correction1DFromCSV – error / validation paths
# ===========================================================================


class TestCorrection1DFromCSV_Validation:

    def test_missing_csv_path_raises(self):
        with pytest.raises(MissingInformationError):
            Correction1DFromCSV(csv_path=None)

    def test_missing_dependant_variable_column_raises(self, tmp_path):
        bad_csv = """\
p_min,p_max,central_value,stat_corr,sys_corr,stat_uncorr,sys_uncorr
0.0,1.0,0.95,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction1DFromCSV(csv_path=str(csv))

    def test_missing_stat_corr_column_raises(self, tmp_path):
        bad_csv = """\
dependant_variable,p_min,p_max,central_value,p_unit,sys_corr,stat_uncorr,sys_uncorr
p,0.0,1.0,0.95,GeV,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad_keys.csv", bad_csv)
        with pytest.raises(InvalidCorrectionTableKey):
            Correction1DFromCSV(csv_path=str(csv))

    def test_missing_bin_edge_and_discrete_column_raises(self, tmp_path):
        # dependant_variable present, but no p_min/p_max and no 'p' discrete column
        bad_csv = """\
dependant_variable,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,0.95,GeV,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "no_edges.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction1DFromCSV(csv_path=str(csv))

    def test_invalid_pdg_format_raises(self, tmp_path):
        bad_csv = """\
dependant_variable,p_min,p_max,central_value,p_unit,stat_corr,sys_corr,stat_uncorr,sys_uncorr,PDG
p,0.0,1.0,0.95,GeV,0.01,0.02,0.005,0.008,521
"""
        csv = _write_csv(tmp_path, "bad_pdg.csv", bad_csv)
        with pytest.raises(ValueError, match="string list format"):
            Correction1DFromCSV(csv_path=str(csv))

    def test_nonexistent_cov_matrix_file_raises(self, tmp_path):
        csv = _write_csv(tmp_path, "binned.csv", BINNED_1D_CSV)
        with pytest.raises(ValueError, match="not found"):
            Correction1DFromCSV(
                csv_path=str(csv),
                cov_matrix_path=str(tmp_path / "does_not_exist.npy"),
            )


# ===========================================================================
# Correction2DFromCSV – construction and methods
# ===========================================================================


@pytest.fixture
def corr2d(tmp_path):
    csv = _write_csv(tmp_path, "2d.csv", BINNED_2D_CSV)
    return Correction2DFromCSV(csv_path=str(csv))


@pytest.fixture
def corr2d_pdg(tmp_path):
    csv = _write_csv(tmp_path, "2d_pdg.csv", PDG_2D_CSV)
    return Correction2DFromCSV(csv_path=str(csv))


class TestCorrection2DFromCSV_Properties:

    def test_N_equals_rows(self, corr2d):
        assert corr2d.N == 4

    def test_dependant_variables_parsed(self, corr2d):
        assert corr2d.dependant_variable_1 == "p"
        assert corr2d.dependant_variable_2 == "cosTheta"

    def test_unit_1_parsed(self, corr2d):
        assert corr2d.unit_1 == "GeV"

    def test_unit_2_empty_string_when_missing(self, corr2d):
        # cosTheta_unit column is empty in our fixture
        assert isinstance(corr2d.unit_2, str)

    def test_central_values_correct(self, corr2d):
        np.testing.assert_allclose(corr2d.central_values, [0.90, 0.92, 0.95, 0.98])

    def test_edge_columns_accessible(self, corr2d):
        assert corr2d._v1_min == "p_min"
        assert corr2d._v1_max == "p_max"
        assert corr2d._v2_min == "cosTheta_min"
        assert corr2d._v2_max == "cosTheta_max"


class TestCorrection2DFromCSV_Iterator:

    def test_iterator_yields_correct_number_of_tuples(self, corr2d):
        bins = list(corr2d.iterator)
        assert len(bins) == 4

    def test_iterator_yields_4_tuples(self, corr2d):
        for b in corr2d.iterator:
            assert len(b) == 4

    def test_iterator_first_row_values(self, corr2d):
        first = next(iter(corr2d.iterator))
        v1min, v1max, v2min, v2max = first
        assert v1min == 0.0
        assert v1max == 1.0
        assert v2min == -1.0
        assert v2max == 0.0

    def test_iterator_is_reusable(self, corr2d):
        # property-based iterator should give same result each time
        assert list(corr2d.iterator) == list(corr2d.iterator)


class TestCorrection2DFromCSV_VisualLabels:

    def test_visual_labels_count(self, corr2d):
        assert len(corr2d.visual_labels) == 4

    def test_visual_labels_contain_both_variable_names(self, corr2d):
        for label in corr2d.visual_labels:
            assert "p" in label
            assert "cosTheta" in label

    def test_visual_labels_first_entry(self, corr2d):
        label = corr2d.visual_labels[0]
        assert "0.0 <= p < 1.0" in label
        assert "-1.0 <= cosTheta < 0.0" in label

    def test_visual_labels_are_all_non_empty_strings(self, corr2d):
        for label in corr2d.visual_labels:
            assert isinstance(label, str) and label.strip()


class TestCorrection2DFromCSV_BuildQueries:

    def test_build_queries_no_prefix(self, corr2d):
        queries = corr2d.build_queries(prefix=None)
        assert len(queries) == 4
        assert "p" in queries[0]
        assert "cosTheta" in queries[0]

    def test_build_queries_with_prefix(self, corr2d):
        queries = corr2d.build_queries(prefix="pi0")
        assert all("pi0_p" in q for q in queries)
        assert all("pi0_cosTheta" in q for q in queries)

    def test_build_queries_bin_operators(self, corr2d):
        for q in corr2d.build_queries():
            assert "<=" in q and "<" in q

    def test_build_queries_with_pdg(self, corr2d_pdg):
        queries = corr2d_pdg.build_queries(prefix=None)
        assert all("PDG ==" in q for q in queries)


class TestCorrection2DFromCSV_Uncertainties:

    def test_uncertainties_populated(self, corr2d):
        assert len(corr2d.uncertainties) >= 1

    def test_total_error_length(self, corr2d):
        assert len(corr2d.total_error) == corr2d.N

    def test_total_error_positive(self, corr2d):
        assert np.all(corr2d.total_error > 0)

    def test_explicit_cov_2d(self, tmp_path):
        csv = _write_csv(tmp_path, "2d.csv", BINNED_2D_CSV)
        cov = np.diag([0.01**2] * 4)
        npy = _write_npy(tmp_path, "cov2d.npy", cov)
        corr = Correction2DFromCSV(csv_path=str(csv), cov_matrix_path=str(npy))
        assert "explicit_covariance" in corr.uncertainties
        np.testing.assert_allclose(corr.total_error, np.sqrt(np.diag(cov)), rtol=1e-6)


class TestCorrection2DFromCSV_Validation:

    def test_missing_dependant_variable_columns_raises(self, tmp_path):
        bad_csv = """\
p_min,p_max,cosTheta_min,cosTheta_max,central_value,stat_corr,sys_corr,stat_uncorr,sys_uncorr
0.0,1.0,-1.0,0.0,0.90,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad_2d.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction2DFromCSV(csv_path=str(csv))

    def test_missing_bin_edge_column_raises(self, tmp_path):
        bad_csv = """\
dependant_variable_1,dependant_variable_2,p_min,p_max,central_value,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,cosTheta,0.0,1.0,0.90,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad_edges_2d.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction2DFromCSV(csv_path=str(csv))


# ===========================================================================
# Correction3DFromCSV – construction and methods
# ===========================================================================


@pytest.fixture
def corr3d(tmp_path):
    csv = _write_csv(tmp_path, "3d.csv", BINNED_3D_CSV)
    return Correction3DFromCSV(csv_path=str(csv))


class TestCorrection3DFromCSV_Properties:

    def test_N_equals_rows(self, corr3d):
        assert corr3d.N == 2

    def test_dependant_variables_parsed(self, corr3d):
        assert corr3d.dependant_variable_1 == "p"
        assert corr3d.dependant_variable_2 == "cosTheta"
        assert corr3d.dependant_variable_3 == "phi"

    def test_central_values(self, corr3d):
        np.testing.assert_allclose(corr3d.central_values, [0.90, 0.92])

    def test_edge_column_names(self, corr3d):
        assert corr3d._v1_min == "p_min"
        assert corr3d._v2_min == "cosTheta_min"
        assert corr3d._v3_min == "phi_min"


class TestCorrection3DFromCSV_Iterator:

    def test_iterator_count(self, corr3d):
        assert len(list(corr3d.iterator)) == 2

    def test_iterator_yields_6_tuples(self, corr3d):
        for b in corr3d.iterator:
            assert len(b) == 6

    def test_iterator_first_row_values(self, corr3d):
        first = next(iter(corr3d.iterator))
        v1min, v1max, v2min, v2max, v3min, v3max = first
        assert v1min == 0.0
        assert v1max == 1.0
        assert v2min == -1.0
        assert v2max == 0.0
        assert v3min == 0.0
        assert v3max == 1.0

    def test_iterator_is_reusable(self, corr3d):
        assert list(corr3d.iterator) == list(corr3d.iterator)


class TestCorrection3DFromCSV_VisualLabels:

    def test_visual_labels_count(self, corr3d):
        assert len(corr3d.visual_labels) == 2

    def test_visual_labels_contain_all_variable_names(self, corr3d):
        for label in corr3d.visual_labels:
            assert "p" in label
            assert "cosTheta" in label
            assert "phi" in label

    def test_visual_labels_first_entry(self, corr3d):
        label = corr3d.visual_labels[0]
        assert "0.0 <= p < 1.0" in label
        assert "-1.0 <= cosTheta < 0.0" in label
        assert "0.0 <= phi < 1.0" in label

    def test_visual_labels_non_empty(self, corr3d):
        for label in corr3d.visual_labels:
            assert isinstance(label, str) and label.strip()


class TestCorrection3DFromCSV_BuildQueries:

    def test_build_queries_no_prefix(self, corr3d):
        queries = corr3d.build_queries(prefix=None)
        assert len(queries) == 2
        assert "p" in queries[0]
        assert "cosTheta" in queries[0]
        assert "phi" in queries[0]

    def test_build_queries_with_prefix(self, corr3d):
        queries = corr3d.build_queries(prefix="ks")
        assert all("ks_p" in q for q in queries)
        assert all("ks_cosTheta" in q for q in queries)
        assert all("ks_phi" in q for q in queries)

    def test_build_queries_bin_operators(self, corr3d):
        for q in corr3d.build_queries():
            assert "<=" in q and "<" in q


class TestCorrection3DFromCSV_Uncertainties:

    def test_uncertainties_populated(self, corr3d):
        assert len(corr3d.uncertainties) >= 1

    def test_total_error_length(self, corr3d):
        assert len(corr3d.total_error) == corr3d.N

    def test_total_error_positive(self, corr3d):
        assert np.all(corr3d.total_error > 0)


class TestCorrection3DFromCSV_Validation:

    def test_missing_third_dependant_variable_raises(self, tmp_path):
        bad_csv = """\
dependant_variable_1,dependant_variable_2,p_min,p_max,cosTheta_min,cosTheta_max,phi_min,phi_max,central_value,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,cosTheta,0.0,1.0,-1.0,0.0,0.0,1.0,0.90,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad_3d.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction3DFromCSV(csv_path=str(csv))

    def test_missing_bin_edge_column_raises(self, tmp_path):
        bad_csv = """\
dependant_variable_1,dependant_variable_2,dependant_variable_3,p_min,p_max,cosTheta_min,cosTheta_max,central_value,stat_corr,sys_corr,stat_uncorr,sys_uncorr
p,cosTheta,phi,0.0,1.0,-1.0,0.0,0.90,0.01,0.02,0.005,0.008
"""
        csv = _write_csv(tmp_path, "bad_edges_3d.csv", bad_csv)
        with pytest.raises(ValueError):
            Correction3DFromCSV(csv_path=str(csv))


# ===========================================================================
# Cross-class: _extend_queries_with_extra_cut (static, on BaseCorrectionFromYaml)
# ===========================================================================


class TestExtendQueriesWithExtraCut:

    def test_appends_extra_condition_to_all_queries(self):
        from sysvar.corrections import BaseCorrectionFromYaml

        queries = ["a > 1", "b < 2"]
        result = BaseCorrectionFromYaml._extend_queries_with_extra_cut(
            queries, "c == 3"
        )
        assert result == ["a > 1 & c == 3", "b < 2 & c == 3"]

    def test_empty_queries_list(self):
        from sysvar.corrections import BaseCorrectionFromYaml

        result = BaseCorrectionFromYaml._extend_queries_with_extra_cut([], "c == 3")
        assert result == []

    def test_single_query(self):
        from sysvar.corrections import BaseCorrectionFromYaml

        result = BaseCorrectionFromYaml._extend_queries_with_extra_cut(
            ["x > 0"], "y == 1"
        )
        assert result == ["x > 0 & y == 1"]


# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_bf_info(
    correlation: str = "fully_correlated",
    extra_cuts: dict | None = None,
    modes: dict | None = None,
):
    """Return a minimal info dict that satisfies CorrectionBF.__post_init__."""
    if modes is None:
        modes = {
            "mode_a": {
                "pdg_live": [0.10, 0.01],
                "decay_dec": 0.08,
                "daughters": [[521, -211]],
                "dmID": "mode_a_id",
            },
            "mode_b": {
                "pdg_live": [0.20, 0.02],
                "decay_dec": 0.18,
                "daughters": [[521, -321]],
                "dmID": [10, 11],  # list → uses `in` operator in query
            },
        }
    return {
        "title": "Test BF Correction",
        "mother_particle": 521,  # B+
        "modes": modes,
        "correlation": correlation,
        "extra_cuts": extra_cuts,
        "dependant_variable": "dmID",
    }


def _make_custom_info(
    n: int = 3,
    cov_matrix: list | None = None,
):
    """Return a minimal info dict for CustomCorrection."""
    info = {
        "dependant_variable": "channel",
        "central_values": [0.95, 1.00, 1.05][:n],
        "query_targets": ["ch_A", "ch_B", "ch_C"][:n],
        "unit": "GeV",
        "title": "Test Custom Correction",
        "uncertainties": {
            "fully_correlated": {
                "sys": [0.02, 0.03, 0.025][:n],
            },
            "uncorrelated": {
                "stat": [0.01, 0.01, 0.01][:n],
            },
        },
    }
    if cov_matrix is not None:
        info["my_cov"] = cov_matrix
    return info


# ===========================================================================
# CorrectionBF fixture
# ===========================================================================


@pytest.fixture
def bf_info():
    return _make_bf_info()


@pytest.fixture
def corr_bf(bf_info):
    """
    CorrectionBF with its YAML loading patched out so no real config files
    are needed. cov_matrix is forced to None (no explicit covariance file).
    """
    with patch("sysvar.corrections.read_yaml", return_value=bf_info), patch(
        "sysvar.corrections.load_covariance_matrix", return_value=None
    ):
        corr = CorrectionBF(systematic="fake_sys", MC_production="fake_prod")
    return corr


# ===========================================================================
# CorrectionBF – _calculate_scaling_ratios
# ===========================================================================


class TestCorrectionBF_ScalingRatios:

    def test_central_values_length_matches_modes(self, corr_bf, bf_info):
        assert len(corr_bf.central_values) == len(bf_info["modes"])

    def test_central_values_are_floats(self, corr_bf):
        assert all(isinstance(v, float) for v in corr_bf.central_values)

    def test_central_values_are_ratios(self, bf_info):
        """central_value[i] ≈ pdg_live[0] / decay_dec."""
        with patch("sysvar.corrections.read_yaml", return_value=bf_info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")

        modes = list(bf_info["modes"].values())
        for i, mode in enumerate(modes):
            expected = mode["pdg_live"][0] / mode["decay_dec"]
            assert pytest.approx(corr.central_values[i], rel=1e-6) == expected

    def test_zero_decay_dec_gives_central_value_one(self):
        """When decay_dec == 0, the safe-divide fallback returns 1 ± 1."""
        info = _make_bf_info(
            modes={
                "mode_zero": {
                    "pdg_live": [0.10, 0.01],
                    "decay_dec": 0.0,
                    "daughters": [[521, -211]],
                    "dmID": "zero",
                }
            }
        )
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")
        assert pytest.approx(corr.central_values[0]) == 1.0

    def test_zero_pdg_gives_central_value_one(self):
        """When pdg_live == 0, the safe-divide fallback returns 1 ± 1."""
        info = _make_bf_info(
            modes={
                "mode_zero": {
                    "pdg_live": [0.0, 0.0],
                    "decay_dec": 0.5,
                    "daughters": [[521, -211]],
                    "dmID": "zero",
                }
            }
        )
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")
        assert pytest.approx(corr.central_values[0]) == 1.0


# ===========================================================================
# CorrectionBF – visual_labels / _create_strings
# ===========================================================================


class TestCorrectionBF_VisualLabels:

    def test_visual_labels_count_matches_modes(self, corr_bf, bf_info):
        # one label per daughter set across all modes
        total_daughter_sets = sum(
            len(mode["daughters"]) for mode in bf_info["modes"].values()
        )
        assert len(corr_bf.visual_labels) == total_daughter_sets

    def test_visual_labels_are_non_empty_strings(self, corr_bf):
        for label in corr_bf.visual_labels:
            assert isinstance(label, str) and label.strip()

    def test_visual_labels_contain_arrow(self, corr_bf):
        for label in corr_bf.visual_labels:
            assert r"\rightarrow" in label

    def test_visual_labels_wrapped_in_math_mode(self, corr_bf):
        for label in corr_bf.visual_labels:
            assert label.startswith("$") and label.endswith("$")

    def test_visual_labels_multiple_daughter_sets(self):
        """Two daughter sets in a single mode → two labels."""
        info = _make_bf_info(
            modes={
                "mode_multi": {
                    "pdg_live": [0.10, 0.01],
                    "decay_dec": 0.08,
                    "daughters": [[521, -211], [521, -321]],
                    "dmID": "multi",
                }
            }
        )
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")
        assert len(corr.visual_labels) == 2


# ===========================================================================
# CorrectionBF – uncertainties / populate_uncertainties
# ===========================================================================


class TestCorrectionBF_Uncertainties:

    def test_bf_unc_key_exists(self, corr_bf):
        assert "BF_unc" in corr_bf.uncertainties

    def test_total_error_length_matches_N(self, corr_bf):
        assert len(corr_bf.total_error) == corr_bf.N

    def test_total_error_is_positive(self, corr_bf):
        assert np.all(corr_bf.total_error >= 0)

    def test_N_equals_central_values_length(self, corr_bf):
        assert corr_bf.N == len(corr_bf.central_values)

    @pytest.mark.parametrize(
        "correlation", ["fully_correlated", "uncorrelated", "fully_correlated_in_parts"]
    )
    def test_valid_correlation_types_accepted(self, correlation):
        info = _make_bf_info(correlation=correlation)
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")
        assert "BF_unc" in corr.uncertainties

    def test_explicitly_correlated_raises_not_implemented(self):
        info = _make_bf_info(correlation="explicitly_correlated")
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            with pytest.raises(NotImplementedError):
                CorrectionBF(systematic="x", MC_production="y")

    def test_unknown_correlation_raises_value_error(self):
        info = _make_bf_info(correlation="made_up_type")
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            with pytest.raises(ValueError, match="Unkown correlation type"):
                CorrectionBF(systematic="x", MC_production="y")


# ===========================================================================
# CorrectionBF – build_queries
# ===========================================================================


class TestCorrectionBF_BuildQueries:

    def test_returns_list(self, corr_bf):
        assert isinstance(corr_bf.build_queries(), list)

    def test_length_matches_modes(self, corr_bf, bf_info):
        assert len(corr_bf.build_queries()) == len(bf_info["modes"])

    def test_string_dmid_uses_equality(self, corr_bf):
        """mode_a has dmID='mode_a_id' → should use == 'mode_a_id'."""
        queries = corr_bf.build_queries(prefix=None)
        assert queries[0] == "dmID == 'mode_a_id'"

    def test_list_dmid_uses_in(self, corr_bf):
        """mode_b has dmID=[10, 11] → should use `in [10, 11]`."""
        queries = corr_bf.build_queries(prefix=None)
        assert "dmID in [10, 11]" in queries[1]

    def test_prefix_prepended_to_variable(self, corr_bf):
        queries = corr_bf.build_queries(prefix="slow_pi")
        for q in queries:
            assert "slow_pi_dmID" in q

    def test_no_prefix_uses_variable_directly(self, corr_bf):
        queries = corr_bf.build_queries(prefix=None)
        for q in queries:
            assert q.startswith("dmID")

    def test_all_queries_are_non_empty_strings(self, corr_bf):
        for q in corr_bf.build_queries():
            assert isinstance(q, str) and q.strip()

    def test_extra_cuts_injected_when_present(self):
        """extra_cuts in info should be appended to each query."""
        info = _make_bf_info(extra_cuts={"charge": [1, -1]})
        with patch("sysvar.corrections.read_yaml", return_value=info), patch(
            "sysvar.corrections.load_covariance_matrix", return_value=None
        ):
            corr = CorrectionBF(systematic="x", MC_production="y")
        queries = corr.build_queries(prefix=None)
        for q in queries:
            assert "charge" in q

    def test_no_extra_cuts_when_none(self, corr_bf):
        """With extra_cuts=None, queries stay clean."""
        queries = corr_bf.build_queries(prefix=None)
        for q in queries:
            # Should only contain the dmID condition
            assert "charge" not in q


# ===========================================================================
# CorrectionBF – dependant_variable
# ===========================================================================


class TestCorrectionBF_DependantVariable:

    def test_dependant_variable_set_from_info(self, corr_bf):
        assert corr_bf.dependant_variable == "dmID"


# ===========================================================================
# CustomCorrection fixtures
# ===========================================================================


@pytest.fixture
def custom_info():
    return _make_custom_info()


@pytest.fixture
def corr_custom(custom_info):
    return CustomCorrection(info=custom_info)


@pytest.fixture
def corr_custom_with_cov():
    cov = np.diag([0.02**2, 0.03**2, 0.025**2]).tolist()
    info = _make_custom_info(cov_matrix=cov)
    info["my_cov"] = cov
    # load_covariance_matrix reads key_matrix; give it the right key
    return CustomCorrection(info=info)


# ===========================================================================
# CustomCorrection – construction / attributes
# ===========================================================================


class TestCustomCorrection_Properties:

    def test_dependant_variable(self, corr_custom):
        assert corr_custom.dependant_variable == "channel"

    def test_central_values_length(self, corr_custom):
        assert len(corr_custom.central_values) == 3

    def test_central_values_correct(self, corr_custom):
        np.testing.assert_allclose(corr_custom.central_values, [0.95, 1.00, 1.05])

    def test_query_targets_set(self, corr_custom):
        assert corr_custom.query_targets == ["ch_A", "ch_B", "ch_C"]

    def test_unit_set(self, corr_custom):
        assert corr_custom.unit == "GeV"

    def test_title_set(self, corr_custom):
        assert corr_custom.title == "Test Custom Correction"

    def test_N_equals_central_values_length(self, corr_custom):
        assert corr_custom.N == len(corr_custom.central_values)

    def test_cov_matrix_none_when_not_provided(self, corr_custom):
        assert corr_custom.cov_matrix is None


# ===========================================================================
# CustomCorrection – value_edges and value_mids
# ===========================================================================


class TestCustomCorrection_EdgesAndMids:

    def test_value_edges_length(self, corr_custom):
        assert len(corr_custom.value_edges) == corr_custom.N + 1

    def test_value_edges_values(self, corr_custom):
        np.testing.assert_array_equal(corr_custom.value_edges, np.arange(4))

    def test_value_mids_length(self, corr_custom):
        assert len(corr_custom.value_mids) == corr_custom.N

    def test_value_mids_values(self, corr_custom):
        np.testing.assert_array_almost_equal(corr_custom.value_mids, [0.5, 1.5, 2.5])

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_value_edges_for_various_n(self, n):
        info = _make_custom_info(n=n)
        corr = CustomCorrection(info=info)
        assert len(corr.value_edges) == n + 1

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_value_mids_for_various_n(self, n):
        info = _make_custom_info(n=n)
        corr = CustomCorrection(info=info)
        assert len(corr.value_mids) == n


# ===========================================================================
# CustomCorrection – visual_labels
# ===========================================================================


class TestCustomCorrection_VisualLabels:

    def test_visual_labels_count(self, corr_custom):
        assert len(corr_custom.visual_labels) == 3

    def test_visual_labels_contain_variable(self, corr_custom):
        for label in corr_custom.visual_labels:
            assert "channel" in label

    def test_visual_labels_contain_target(self, corr_custom):
        for label, target in zip(corr_custom.visual_labels, corr_custom.query_targets):
            assert target in label

    def test_visual_labels_format(self, corr_custom):
        assert corr_custom.visual_labels[0] == "channel = ch_A"
        assert corr_custom.visual_labels[1] == "channel = ch_B"
        assert corr_custom.visual_labels[2] == "channel = ch_C"

    def test_visual_labels_non_empty_strings(self, corr_custom):
        for label in corr_custom.visual_labels:
            assert isinstance(label, str) and label.strip()


# ===========================================================================
# CustomCorrection – build_queries
# ===========================================================================


class TestCustomCorrection_BuildQueries:

    def test_build_queries_returns_list(self, corr_custom):
        assert isinstance(corr_custom.build_queries(), list)

    def test_build_queries_length_matches_N(self, corr_custom):
        assert len(corr_custom.build_queries()) == corr_custom.N

    def test_build_queries_no_prefix(self, corr_custom):
        queries = corr_custom.build_queries(prefix=None)
        assert queries[0] == "channel == 'ch_A'"
        assert queries[1] == "channel == 'ch_B'"
        assert queries[2] == "channel == 'ch_C'"

    def test_build_queries_with_prefix(self, corr_custom):
        queries = corr_custom.build_queries(prefix="trk")
        assert all("trk_channel ==" in q for q in queries)

    def test_build_queries_targets_quoted(self, corr_custom):
        for q in corr_custom.build_queries():
            # targets are strings → must be single-quoted in the query
            assert "'" in q

    def test_build_queries_all_non_empty(self, corr_custom):
        for q in corr_custom.build_queries():
            assert isinstance(q, str) and q.strip()

    def test_build_queries_default_prefix_is_none(self, corr_custom):
        """Calling with no argument should behave the same as prefix=None."""
        assert corr_custom.build_queries() == corr_custom.build_queries(prefix=None)


# ===========================================================================
# CustomCorrection – uncertainties / populate_uncertainties
# ===========================================================================


class TestCustomCorrection_Uncertainties:

    def test_uncertainties_populated(self, corr_custom):
        assert len(corr_custom.uncertainties) >= 1

    def test_total_error_length(self, corr_custom):
        assert len(corr_custom.total_error) == corr_custom.N

    def test_total_error_positive(self, corr_custom):
        assert np.all(corr_custom.total_error > 0)

    def test_explicit_cov_uncertainty_added(self):
        cov = np.diag([0.02**2, 0.03**2, 0.025**2])
        info = _make_custom_info()
        info["my_cov"] = cov.tolist()
        corr = CustomCorrection(info=info)
        assert "explicit_covariance" in corr.uncertainties

    def test_explicit_cov_total_error_matches_diagonal(self):
        cov = np.diag([0.02**2, 0.03**2, 0.025**2])
        info = _make_custom_info()
        info["my_cov"] = cov.tolist()
        corr = CustomCorrection(info=info)
        expected = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(corr.total_error, expected, rtol=1e-6)

    def test_add_duplicate_uncertainty_raises(self, corr_custom):
        from sysvar.uncertainties import FullyCorrelatedUncertainty

        existing_name = next(iter(corr_custom.uncertainties))
        with pytest.raises(UncertaintyWithSameNameExists):
            corr_custom.add_uncertainty(
                unc_name=existing_name,
                unc_values=[0.01] * corr_custom.N,
                unc_obj=FullyCorrelatedUncertainty,
            )

    def test_empty_uncertainties_total_error_raises(self, corr_custom):
        corr_custom.uncertainties = {}
        with pytest.raises(ValueError):
            _ = corr_custom.total_error


# ===========================================================================
# CustomCorrection – edge cases
# ===========================================================================


class TestCustomCorrection_EdgeCases:

    def test_single_bin(self):
        info = _make_custom_info(n=1)
        corr = CustomCorrection(info=info)
        assert corr.N == 1
        assert len(corr.visual_labels) == 1
        assert len(corr.build_queries()) == 1
        assert len(corr.value_edges) == 2
        assert len(corr.value_mids) == 1

    def test_many_bins(self):
        n = 10
        info = {
            "dependant_variable": "ch",
            "central_values": list(np.linspace(0.9, 1.1, n)),
            "query_targets": [f"ch_{i}" for i in range(n)],
            "unit": "",
            "title": "large",
            "uncertainties": {
                "uncorrelated": {"stat": [0.01] * n},
            },
        }
        corr = CustomCorrection(info=info)
        assert corr.N == n
        assert len(corr.value_edges) == n + 1
        assert len(corr.value_mids) == n
        assert len(corr.visual_labels) == n
        assert len(corr.build_queries()) == n

    def test_empty_unit_string(self):
        info = _make_custom_info()
        info["unit"] = ""
        corr = CustomCorrection(info=info)
        assert corr.unit == ""
