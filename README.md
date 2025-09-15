# Deep-Learning-Comparisons
Build mini-batch deep learning from scratch (Python, NumPy) and compare experiments between differing weight initializations, learning rates, and batch sizes.

# Mini-batch Gradient Descent

## Project Structure
MBGD/
│
├── data/ # Linear and logistic data generators
├── models/ # Linear and logistic regression models
├── optim/ # Gradient descent implementation
├── experiments/ # Scripts for all 3 experiments
├── utils/ # Metrics & plotting helpers
├── tests/ # Unit tests
├── main.py # Orchestrator script
└── README.md

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Unittest (for tests)

## Running Experiments
```bash
python main.py
```

This will execute all experiments and generate plots in the project directory.

## Running Tests
```bash
python -m unittest discover tests
```

## Notes
All results are reproducible using the fixed random seeds in data generation.
Mini-batch gradient descent is implemented in optim/gradient_descent.py.
Loss curves for training and testing are generated and saved as PNG files.
Default seed is set to `42` because that is the answer to the meaning of life, the universe, and everything.

## Algorithms
Data Generation
- Uniform sampling
- Gaussian sampling
- Shuffling

Model Prediction
- Linear regression
  - y_hat = Xw
- Logistic regression
  - p_hat = sigmoid(Xw)
  - sigmoid(z) = 1 / ( 1 + e^(-z) )

Loss Functions
- Mean squared error (MSE) for linear regression
  - L = (1 / B) * ||Xw - y||^2
- Cross-entropy loss for logistic regression
  - L = -(1 / B) * sum( (y * log(p_hat)) - ((1 - y) * log(1 - p_hat)) )

Gradient Computation
- Linear regression
  - gradient_w(L) = (1 / B) * X_transpose * (Xw - y)
- Logistic regression
  - gradient_w(L) = (1 / B) * X_transpose * (p_hat - y)

Optimization
- Mini-batch gradient descent
  - Choose batch size B
  - Each iteration, sample a batch (X_b, y_b)
  - Compute the gradient: gradient_w(L) = (X_b, y_b)
  - Update weights: w <- w - eta * gradient_w(L)
**Special Cases: batch-size 1 is stochastic gradient descent, and batch-size *all* is batch gradient descent**

Training Enhancements
- Early stopping
  - When loss change < tolerance (tol)

Experiments
- Experiment 1
  - Zero vs uniform random weight initialization
- Experiment 2
  - Gradient descent with eta one of 0.01, 0.05, 0.1, or 0.2
- Experiment 3
  - Differing batch sizes as one of 2, 32, 256, 512, 4096, or 8192

Utility
- Metrics
  - Compute MSE and cross-entropy on train/test sets in training
- Plotting
  - *Generative loss vs iteration* plots with Matplotlib
- CSV Logging
  - Hyperparameters and final metrics
  

## Math
