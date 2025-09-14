import unittest
import numpy as np
from utils.metrics import mean_squared_error, cross_entropy_loss, accuracy

class TestMetrics(unittest.TestCase):
    def test_mse(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), (0 + 0.25 + 0.25) / 3)

    def test_cross_entropy_perfect(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0.0, 1.0])
        loss = cross_entropy_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_cross_entropy_symmetric(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0.5, 0.5])
        loss = cross_entropy_loss(y_true, y_pred)
        # -[0*log(.5) + 1*log(.5)] = log(2)
        self.assertAlmostEqual(loss, np.log(2), places=6)

    def test_accuracy(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.4, 0.2])
        acc = accuracy(y_true, y_pred, threshold=0.5)
        self.assertEqual(acc, 0.75)

if __name__ == "__main__":
    unittest.main()