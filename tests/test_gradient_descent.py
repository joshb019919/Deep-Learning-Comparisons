import unittest 
import numpy as np
from optim.gradient_descent import gradient_descent
from models.linear_regression import predict as lin_predict, loss as lin_loss, gradient as lin_grad
from models.logistic_regression import predict as log_predict, loss as log_loss, gradient as log_grad
from data.linear_data import generate_linear_data
from data.logistic_data import generate_logistic_data

class TestGradientDescent(unittest.TestCase):
    def test_linear_regression_convergence(self) -> None:
        X_train, y_train, _, _ = generate_linear_data(200, 50, seed=0)
        w_init = np.zeros(X_train.shape[1])
        _, train_losses, _ = gradient_descent(
            X_train, y_train,
            lin_predict, lin_loss, lin_grad,
            w_init, lr=0.1, batch_size=32,
            max_iters=200
        )
        self.assertLess(train_losses[-1], train_losses[0])  # loss decreased

    def test_logistic_regression_convergence(self) -> None:
        X_train, y_train, _, _ = generate_logistic_data(200, seed=0)
        w_init = np.zeros(X_train.shape[1])
        _, train_losses, _ = gradient_descent(
            X_train, y_train,
            log_predict, log_loss, log_grad,
            w_init, lr=0.1, batch_size=32,
            max_iters=200
        )
        self.assertLess(train_losses[-1], train_losses[0])  # loss decreased

    def test_batch_size_full_vs_sgd(self) -> None:
        X_train, y_train, _, _ = generate_linear_data(100, 50, seed=0)
        w_init = np.zeros(X_train.shape[1])

        # Full batch
        _, losses_batch, _ = gradient_descent(
            X_train, y_train, lin_predict, lin_loss, lin_grad,
            w_init, lr=0.1, batch_size=len(X_train), max_iters=50
        )

        # Stochastic (batch_size=1)
        _, losses_sgd, _ = gradient_descent(
            X_train, y_train, lin_predict, lin_loss, lin_grad,
            w_init, lr=0.1, batch_size=1, max_iters=200
        )

        self.assertLess(losses_batch[-1], losses_batch[0])
        self.assertLess(losses_sgd[-1], losses_sgd[0])

if __name__ == "__main__":
    unittest.main()
