import unittest
import numpy as np
from sysvar.corrections import Correction


class TestCorrection(unittest.TestCase):
    """
    A test case for the Correction class.
    """

    def setUp(self):
        """
        Set up a sample Correction object for testing.
        """
        values = np.array([1, 2, 3])
        lower_bounds = np.array([0, 1, 2])
        upper_bounds = np.array([2, 3, 4])
        self.correction = Correction(values, lower_bounds, upper_bounds)

    def test_build_strings(self):
        """
        Test the build_strings method of the Correction class.

        This method checks whether the build_strings method generates the expected list of strings.
        """
        expected_strings = ["[ 0 - 2 ]", "[ 1 - 3 ]", "[ 2 - 4 ]"]
        result_strings = self.correction.build_strings()
        self.assertEqual(result_strings, expected_strings)

    def test_values(self):
        """
        Test the values attribute of the Correction class.

        This method verifies that the values attribute of the Correction object matches the expected values.
        """
        expected_values = np.array([1, 2, 3])
        result_values = self.correction.values
        np.testing.assert_array_equal(result_values, expected_values)

    def test_lower_bounds(self):
        """
        Test the lower_bounds attribute of the Correction class.

        This method checks that the lower_bounds attribute of the Correction object matches the expected values.
        """
        expected_lower_bounds = np.array([0, 1, 2])
        result_lower_bounds = self.correction.lower_bounds
        np.testing.assert_array_equal(result_lower_bounds, expected_lower_bounds)

    def test_upper_bounds(self):
        """
        Test the upper_bounds attribute of the Correction class.

        This method verifies that the upper_bounds attribute of the Correction object matches the expected values.
        """
        expected_upper_bounds = np.array([2, 3, 4])
        result_upper_bounds = self.correction.upper_bounds
        np.testing.assert_array_equal(result_upper_bounds, expected_upper_bounds)


if __name__ == "__main__":
    unittest.main()
