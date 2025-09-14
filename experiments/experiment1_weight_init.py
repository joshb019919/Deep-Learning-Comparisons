# Skeleton: load data, initialize weights, call gradient descent, plot results
import numpy as np
from data.linear_data import generate_linear_data
from data.logistic_data import generate_logistic_data
from models.linear_regression import predict as lin_predict, loss as lin_loss, gradient as lin_grad
from models.logistic_regression import predict as log_predict, loss as log_loss, gradient as log_grad
from optim.gradient_descent import gradient_descent
from utils.plotting import plot_loss_curves
from utils.csv_logger import save_metrics_to_csv


def run_experiment1_linear(*generates) -> None:
    """Run the linear regression experiment on differing weight initializations."""
    X_train, y_train, X_test, y_test = generate_linear_data()
    metrics = []
    
    for init_name, w_init in {
        "zeros": np.zeros(X_train.shape[1]),
        "uniform": np.random.rand(X_train.shape[1])
    }.items():
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            lin_predict, lin_loss, lin_grad, w_init,
            X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Linear Regression - Init: {init_name}",
                ylabel="MSE",
                save_path=f"reports/figures/linear_init_{init_name}.png"
            )

        if "reports":
            metrics.append({
                "model": "linear",
                "init": init_name,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/0experiment1_linear_metrics.csv", 
                metrics, 
                fieldnames=["model", "init", "final_train_loss", "final_test_loss"]
            )


def run_experiment1_logistic(*generates) -> None:
    """Run the logistic regression experiment on differing weight initializations."""
    X_train, y_train, X_test, y_test = generate_logistic_data()
    metrics = []

    for init_name, w_init in {
        "zeros": np.zeros(X_train.shape[1]),
        "uniform": np.random.rand(X_train.shape[1])
    }.items():
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            log_predict, log_loss, log_grad, w_init,
            X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Logistic Regression - Init: {init_name}",
                ylabel="Cross Entropy",
                save_path=f"reports/figures/logistic_init_{init_name}.png"
            )

        if "reports":
            metrics.append({
                "model": "logistic",
                "init": init_name,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/0experiment1_logistic_metrics.csv", 
                metrics, 
                fieldnames=["model", "init", "final_train_loss", "final_test_loss"]
            )

if __name__ == "__main__":
    run_experiment1_linear()
    run_experiment1_logistic()
