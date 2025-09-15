"""Run linear and logistic regression experiments to compare deep
learning techniques for weight initialization, learning rates, and
batch sizes.

Different weights are all zeros or all uniformly distributed.
Different learning rates are 0.01, 0.05, 0.1, and 0.2.
Different batch sizes are 2, 32, 256, 512, 4096, and 8192.

Experiments can be run to generate reports, figures, or both.
Uncomment the experiment that generates what you want and comment the
rest out (for speed).
"""

from experiments import (
    experiment1_weight_init as exp1, 
    experiment2_learning_rate as exp2, 
    experiment3_batch_size as exp3
)

def main() -> None:
    print("Running Experiment 1: Weight Initialization")
    # exp1.run_experiment1_linear("reports", "figures")
    # exp1.run_experiment1_logistic("reports", "figures")
    # exp1.run_experiment1_linear("figures")
    # exp1.run_experiment1_logistic("figures")
    exp1.run_experiment1_linear("reports")
    exp1.run_experiment1_logistic("reports")
        
    print("Running Experiment 2: Learning Rate Comparison")
    # exp2.run_experiment2_linear("reports", "figures")
    # exp2.run_experiment2_logistic("reports", "figures")
    # exp2.run_experiment2_linear("figures")
    # exp2.run_experiment2_logistic("figures")
    exp2.run_experiment2_linear("reports")
    exp2.run_experiment2_logistic("reports")

    print("Running Experiment 3: Batch Size Comparison")
    # exp3.run_experiment3_linear("reports", "figures")
    # exp3.run_experiment3_logistic("reports", "figures")
    # exp3.run_experiment3_linear("figures")
    # exp3.run_experiment3_logistic("figures")
    exp3.run_experiment3_linear("reports")
    exp3.run_experiment3_logistic("reports")

    print("All experiments completed. Check generated plots in project directory.")

if __name__ == "__main__":
    main()