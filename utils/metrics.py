import numpy as np
from typing import Callable
from numpy import float64, ndarray


def mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float64:
    """Compute the mean squared error."""
    return np.mean((y_true - y_pred) ** 2 )


def cross_entropy_loss(y_true: ndarray, y_pred: ndarray, eps=1e-12) -> float64:
    """Compute binary cross entropy loss.
    
    Args:
        y_true: actual labels, shape (N,)
        y_pred: predicted probabilities in [0,1], shape (N,)
        eps: small constant to avoid log(0)
    """

    y_pred = np.clip(y_pred, eps, 1-eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred))
    return loss


def accuracy(y_true: ndarray, y_pred_probs: ndarray, threshold=0.5) -> float64:
    """Compute classification accuracy for logistic regression."""
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_true == y_pred_labels)
