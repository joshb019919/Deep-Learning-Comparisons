import numpy as np
from data.linear_data import generate_linear_data
from data.logistic_data import generate_logistic_data
from models.linear_regression import predict as lin_predict, loss as lin_loss, gradient as lin_grad
from models.logistic_regression import predict as log_predict, loss as log_loss, gradient as log_grad
from optim.gradient_descent import gradient_descent
from utils.plotting import plot_loss_curves
from utils.csv_logger import save_metrics_to_csv

def run_experiment3_linear(*generates) -> None:
    """Run the linear regression experiment on differing batch sizes."""
    X_train, y_train, X_test, y_test = generate_linear_data()
    w_init = np.zeros(X_train.shape[1])
    metrics = []

    for batch_size in [2, 32, 256, 512, 4096, 8192]:
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            lin_predict, lin_loss, lin_grad, w_init,
            batch_size=batch_size, X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Linear Regression - Batch Size {batch_size}",
                ylabel="MSE",
                save_path=f"reports/figures/linear_bs_{batch_size}.png"
            )

        if "reports":
            metrics.append({
                "model": "linear",
                "batch_size": batch_size,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/0experiment3_linear_metrics.csv", 
                metrics, 
                fieldnames=["model", "batch_size", "final_train_loss", "final_test_loss"]
            )


def run_experiment3_logistic(*generates) -> None:
    """Run the logistic regression experiment on differing batch sizes."""
    X_train, y_train, X_test, y_test = generate_logistic_data()
    w_init = np.zeros(X_train.shape[1])
    metrics = []

    for batch_size in [2, 32, 256, 512, 4092, 8196]:
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            log_predict, log_loss, log_grad, w_init,
            batch_size=batch_size, X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Logistic Regression - Batch Size {batch_size}",
                ylabel="Cross Entropy",
                save_path=f"reports/figures/logistic_bs_{batch_size}.png"
            )

        if "reports":
            metrics.append({
                "model": "logistic",
                "batch_size": batch_size,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/0experiment3_logistic_metrics.csv", 
                metrics, 
                fieldnames=["model", "batch_size", "final_train_loss", "final_test_loss"]
            )

    
if __name__ == "__main__":
    run_experiment3_linear()
    run_experiment3_logistic()