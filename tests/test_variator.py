import unittest

import numpy as np

from sysvar.corrections import Correction
from sysvar.uncertainties import FullyCorrelatedUncertainty
from sysvar.variations import Variator, UncertaintyWithSameNameExists


class TestVariatorAddUncertainty(unittest.TestCase):
    """
    A test case for the add_uncertainty method of the Variator class.
    """

    def setUp(self):
        """
        Set up sample data for testing.
        """
        values = np.array([1, 2, 3])
        lower_bounds = np.array([0, 1, 2])
        upper_bounds = np.array([2, 3, 4])
        correction = Correction(values, lower_bounds, upper_bounds)
        self.variator = Variator(correction)

    def test_add_uncertainty_success(self):
        """
        Test adding an uncertainty to the Variator.

        This test checks whether the add_uncertainty method successfully adds an Uncertainty object to the Variator.
        """
        name = "TestUncertainty"
        errors = np.array([0.1, 0.2, 0.3])
        uncertainty = FullyCorrelatedUncertainty(self.variator.correction, name, errors)
        self.variator.add_uncertainty(uncertainty)
        self.assertIn(name, self.variator.uncertainties)

    def test_add_uncertainty_duplicate_name(self):
        """
        Test adding an uncertainty with a duplicate name to the Variator.

        This test checks whether adding an FullyCorrelatedUncertainty with the same name as an existing one raises the appropriate exception.
        """
        name = "TestFullyCorrelatedUncertainty"
        errors = np.array([0.1, 0.2, 0.3])
        uncertainty1 = FullyCorrelatedUncertainty(
            self.variator.correction, name, errors
        )
        uncertainty2 = FullyCorrelatedUncertainty(
            self.variator.correction, name, errors
        )
        self.variator.add_uncertainty(uncertainty1)

        with self.assertRaises(UncertaintyWithSameNameExists):
            self.variator.add_uncertainty(uncertainty2)


if __name__ == "__main__":
    unittest.main()
