import numpy as np
from typing import Callable, List, Optional, Tuple
from numpy import float64, ndarray

def gradient_descent(X: ndarray, y: ndarray, 
                     loss_fn: Callable[[ndarray, ndarray, ndarray], float64], 
                     grad_fn: Callable[[ndarray, ndarray, ndarray], ndarray],
                     w_init: ndarray, lr=0.1, batch_size=32, max_iters=2000, 
                     tol=1e-8, shuffle=True, X_val: Optional[ndarray]=None, 
                     y_val: Optional[ndarray]=None
    ) -> Tuple[ndarray, List[float64], List[float64]]:
    """Generic mini-batch gradient descent.
    
    Args:
        X: input features
        y: targets
        predict_fn: model prediction function
        loss_fn: loss function
        grad_fn: gradient function
        w_init: initial weights
        lr: learning rate
        batch_size: mini-batch size
        max_iters: maximum number of iterations
        tol: tolerance for early stopping
        shuffle: whether to shuffle data each epoch

    Returns:
        w: optimized weights
        loss_history: list of training losses per iteration
    """
    # Skeleton: implement mini-batch updates, early stopping, loss tracking
    N, D = X.shape
    w = w_init.copy()
    train_losses: List[float64] = []
    val_losses: List[float64] = []

    for it in range(max_iters):
        if shuffle:
            idx = np.random.permutation(N)
            X,y = X[idx], y[idx]
    
        # Mini-batch loop
        for start in range(0, N, batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            grad = grad_fn(X_batch, y_batch, w)
            w -= lr * grad

        # Track losses
        train_loss = loss_fn(X, y, w)
        train_losses.append(train_loss)

        if X_val is not None and y_val is not None:
            val_loss = loss_fn(X_val, y_val, w)
            val_losses.append(val_loss)

        # Early stopping
        if it > 0:
            rel_change = abs(train_losses[-2] - train_loss) / (abs(train_losses[-2]) + 1e-12)
            if rel_change < tol:
                break
    
    return w, train_losses, val_losses
