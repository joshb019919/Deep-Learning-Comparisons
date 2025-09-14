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

## Math
