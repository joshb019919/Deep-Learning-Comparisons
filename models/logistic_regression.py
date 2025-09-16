import numpy as np
from numpy import float64, ndarray


def sigmoid(z: ndarray) -> ndarray:
    """Map all real-valued z to the range (0,1)."""
    z = np.clip(z, -709.7, 36.7)
    return 1/(1 + np.exp(-z))


def predict(X: ndarray, w: ndarray) -> ndarray:
    """Compute predicted probabilities for logistic regression.
    
    Args:
        X: input feature matrix, shape (B,D)
        w: input weight vector, shape (D,)

    Returns:
        Predicted probabilities vector, shape (B,)
    """
    return sigmoid(X @ w)


def loss(X: ndarray, y: ndarray, w: ndarray) -> float64:
    """Compute the binary cross-entropy loss.
    
    Args:
        X: input training matrix, shape (B,D)
        y: labels, shape (B,)
        w: weights, shape (D,)

    Returns:
        Scalar loss of the output
    """
    y_hat = predict(X, w)

    # Clip probabilities for numerical stability
    # Means there are no true 0's and no true 1's
    # Anything < 0.00000001 becomes 0.00000001
    # Anything > 0.99999999 becomes 0.99999999
    eps = 1e-8
    y_hat = np.clip(y_hat, eps, 1-eps)
    return -np.mean((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)))


def gradient(X: ndarray, y: ndarray, w: ndarray) -> ndarray:
    """Compute gradient of cross-entropy with respect to weights.
    
    Args:
        X: input feature matrix, shape (B,D)
        y: test labels, shape (B,)
        w: weights, shape (D,)

    Return:
        Gradient, shape (D,)
    """
    y_hat = predict(X,w)
    return X.T @ (y_hat - y) / X.shape[0]
