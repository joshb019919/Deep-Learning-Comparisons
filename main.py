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