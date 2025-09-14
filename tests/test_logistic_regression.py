import unittest
import numpy as np
from models.logistic_regression import predict, sigmoid, gradient, loss


class TestLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        # Tiny testing dataset
        self.X = np.array([
            [0.5, 1.0], [1.5, -0.5]])
        self.y = np.array([1, 0])
        self.w = np.array([0.1, -0.2])

    def test_sigmoid_range(self) -> None:
        z = np.array([-1000, 0, 1000])
        s = sigmoid(z)
        self.assertTrue(np.all(s > 0))
        self.assertTrue(np.all(s < 1))

    def test_predict_shape(self) -> None:
        p = predict(self.X, self.y)
        self.assertEqual(p.shape, (2,))
        self.assertTrue(np.all(p >= 0) and np.all(p <= 1))

    def test_loss_nonnegative(self) -> None:
        l = loss(self.X, self.y, self.w)
        self.assertGreaterEqual(l, 0)

    def test_gradient_shape(self) -> None:
        g = gradient(self.X, self.y, self.w)
        self.assertEqual(g.shape, self.w.shape)

    def test_gradient_numerical_check(self) -> None:
        # finite difference check
        epsilon = 1e-5
        g_analytical = gradient(self.X, self.y, self.w)
        g_numerical = np.zeros_like(self.w)
        for i in range(len(self.w)):
            w_plus = self.w.copy()
            w_plus[i] += epsilon
            w_minus = self.w.copy()
            w_minus[i] -= epsilon
            g_numerical[i] = (loss(self.X, self.y, w_plus) - loss(self.X, self.y, w_minus)) / (2 * epsilon)
        np.testing.assert_allclose(g_analytical, g_numerical, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
