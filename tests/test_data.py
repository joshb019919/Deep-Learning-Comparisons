import unittest
import numpy as np
from typing import Tuple
from data.linear_data import generate_linear_data
from data.logistic_data import generate_logistic_data

class TestGenerateData(unittest.TestCase):
    def test_linear_shapes(self) -> None:
        X_train, y_train, X_test, y_test = generate_linear_data(100, 50, seed=42)
        self.assertEqual(X_train.shape, (100,4))
        self.assertEqual(X_test.shape, (50,4))
        self.assertEqual(y_train.shape, (100,))
        self.assertEqual(y_test.shape, (50,))

    def test_linear_reproducibility(self) -> None:
        X1, y1, _, _ = generate_linear_data(10, 10, seed=42)
        X2, y2, _, _ = generate_linear_data(10, 10, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_linear_variance(self) -> None:
        X_train, _, _, _ = generate_linear_data(1000, 1000, seed=42)
        var: float = X_train.var()
        self.assertTrue(0.05 < var < 0.3, "Variance sanity check failed!!")

    def test_logistic_shapes(self) -> None:
        X_train, y_train, X_test, y_test = generate_logistic_data(100, seed=42)
        self.assertEqual(X_train.shape, (200,4))
        self.assertEqual(y_train.shape, (200,))
        self.assertEqual(X_test.shape, (200,4))
        self.assertEqual(y_test.shape, (200,))

    def test_logistic_labels(self) -> None:
        _, y_train, _, y_test = generate_logistic_data(50, seed=42)
        self.assertTrue(set(np.unique(y_train)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(y_test)).issubset({0.0, 1.0}))

    def test_logistic_reproducibility(self) -> None:
        X1, y1, _, _ = generate_logistic_data(10, seed=42)
        X2, y2, _, _ = generate_logistic_data(10, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    
if __name__ == "__main__":
    unittest.main()
