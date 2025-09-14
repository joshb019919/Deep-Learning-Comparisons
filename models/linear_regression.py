from numpy import ndarray

def predict(X: ndarray, w: ndarray):
    """Predict outputs for linear regression."""
    return X @ w

def loss(X: ndarray, y: ndarray, w: ndarray):
    """Compute mean squared error (MSE) loss."""
    return ((X @ w - y)**2).mean()

def gradient(X: ndarray, y: ndarray, w: ndarray):
    """Compute gradient of MSE with respect to weights."""
    return 2 * X.T @ (X @ w - y) / X.shape[0]
