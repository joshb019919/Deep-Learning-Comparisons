import numpy as np
from numpy import ndarray

def generate_linear_data(n_train=8196, n_test=8196, seed=42
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Generate synthetic data for linear regression.

    **Features are from the distribution**:
        x ~ N[0, 1]^4

    Args:
        n_train: number of training samples
        n_test: number of testing samples
        seed: random seed for reproducibility

    Returns:
        X_train: training features, shape (n_train, 4)
        y_train: training labels, shape (n_train,)
        X_test: testing features, shape (n_test, 4)
        y_test: testing labels, shape (n_test,)
    """

    np.random.seed(seed)
    d = 4
    w_true = np.array([5, -2, 3.5, -1.5])
    X_train = np.random.uniform(0, 1, (n_train, d))
    X_test = np.random.uniform(0, 1, (n_test, d))
    y_train = X_train @ w_true + np.random.normal(0, 5, n_train)
    y_test = X_test @ w_true + np.random.normal(0, 5, n_test)
    return X_train, y_train, X_test, y_test
