import numpy as np
from numpy import ndarray
from typing import Tuple

def generate_logistic_data(n_per_class=4096, seed=42
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Generate synthetic data for binary classification (logistic 
    regression).
    
    Class 0: x ~ N([1,1,1,1], I)
    Class 1: x ~ N([0.5,0.5,0.5,0.5] I)

    Args:
        n_per_class: number of training/testing samples per class
        seed: gets rid of randomness for reproducibility

    Return:
        X_train: training features, shape (2*n_per_class, 4)
        X_test: training labels (0 or 1), shape (2*n_per_class,)
        y_train: testing features, shape (2*n_per_class, 4)
        y_test: testing labels (0 or 1), shape (2*n_per_class,)
    """

    np.random.seed(seed)
    d = 4
    mean0 = np.ones(d)
    mean1 = np.full(d, 0.5)
    cov = np.eye(d)

    # Training samples (feature data)
    X_train0 = np.random.multivariate_normal(mean0, cov, n_per_class)
    X_train1 = np.random.multivariate_normal(mean1, cov, n_per_class)
    y_train0 = np.zeros(n_per_class)
    y_train1 = np.ones(n_per_class)

    # Testing samples (data labels)
    X_test0 = np.random.multivariate_normal(mean0, cov, n_per_class)
    X_test1 = np.random.multivariate_normal(mean1, cov, n_per_class)
    y_test0 = np.zeros(n_per_class)
    y_test1 = np.ones(n_per_class)

    # Combine into one collection, each
    X_train = np.vstack([X_train0, X_train1])
    y_train = np.concatenate([y_train0, y_train1])
    X_test = np.vstack([X_test0, X_test1])
    y_test = np.concatenate([y_test0, y_test1])

    # Shuffle training data per epoch
    # Docs say to use np.random.Generator.permutation instead of the following
    # Might update to that, later, after getting impl working
    idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test
    