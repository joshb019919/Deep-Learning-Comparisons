import numpy as np
from data.linear_data import generate_linear_data
from data.logistic_data import generate_logistic_data
from models.linear_regression import predict as lin_predict, loss as lin_loss, gradient as lin_grad
from models.logistic_regression import predict as log_predict, loss as log_loss, gradient as log_grad
from optim.gradient_descent import gradient_descent
from utils.plotting import plot_loss_curves
from utils.csv_logger import save_metrics_to_csv


def run_experiment2_linear(*generates) -> None:
    """Run the linear regression experiment on differing learning rates."""
    X_train, y_train, X_test, y_test = generate_linear_data()
    w_init = np.zeros(X_train.shape[1])
    metrics = []

    for lr in [0.01, 0.05, 0.1, 0.2]:
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            lin_predict, lin_loss, lin_grad, w_init,
            lr=lr, X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Linear Regression - Learning Rate {lr}",
                ylabel="MSE",
                save_path=f"reports/figures/linear_lr_{lr}.png"
            )

        if "reports":
            metrics.append({
                "model": "linear",
                "learning_rate": lr,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/experiment2_linear_metrics.csv", 
                metrics, 
                fieldnames=["model", "learning_rate", "final_train_loss", "final_test_loss"]
            )

    
def run_experiment2_logistic(*generates) -> None:
    """Run the logistic regression experiment on differing learning rates."""
    X_train, y_train, X_test, y_test = generate_logistic_data()
    w_init = np.zeros(X_train.shape[1])
    metrics=[]

    for lr in [0.01, 0.05, 0.1, 0.2]:
        w, train_losses, test_losses = gradient_descent(
            X_train, y_train,
            log_predict, log_loss, log_grad, w_init,
            lr=lr, tol=1e-6, X_val=X_test, y_val=y_test
        )

        if "figures":
            plot_loss_curves(
                train_losses, test_losses,
                title=f"Logistic Regression - Learning Rate {lr}",
                ylabel="Cross-Entropy",
                save_path=f"reports/figures/logistic_lr_{lr}.png"
            )

        if "reports":
            metrics.append({
                "model": "logistic",
                "learning_rate": lr,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1]
            })

            save_metrics_to_csv(
                "reports/experiment2_logistic_metrics.csv", 
                metrics, 
                fieldnames=["model", "learning_rate", "final_train_loss", "final_test_loss"]
            )


if __name__ == "__main__":
    run_experiment2_linear()
    run_experiment2_logistic()
