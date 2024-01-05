# Evolutionary Optimization of MLP Neural Networks

## Overview
This project focuses on the comparative performance analysis of evolutionary optimization techniques—Genetic Algorithms (GA), Evolution Strategies (ES), and Stochastic Gradient Descent (SGD)—for training Multilayer Perceptron (MLP) neural networks. The experiment leverages synthetic data to evaluate the optimizers' performance across various noise levels and function complexities.

## Project Structure

```
mlp_regression_project

├── notebooks/
│ └── plot_synthetic_data.ipynb
│
├── output/
│
├── plots/
│ ├── EDA/
│ └── results/
│ ├── data_by_condition/
│ └── metrics/
│
├── results/
│ └── data/
│
├── src/
│ ├── data/
│ │ ├── init.py
│ │ └── load_synthetic_data.py
│ │
│ ├── models/
│ │ ├── init.py
│ │ └── mlp.py
│ │
│ ├── optimization/
│ │ ├── init.py
│ │ ├── evolution_strategies.py
│ │ ├── genetic_algorithm.py
│ │ └── stochastic_gradient_descent.py
│ │
│ └── utils/
│ ├── init.py
│ └── visualization.py
│
├── venv/
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Prerequisites
To run the code in this project, you need to have Python 3.7 installed along with the following libraries:
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- DEAP
- Pandas
- Seaborn

Additionally, the use of PyCharm as IDE is recommended.

## Installation
1. If not installed, clone the repository:
```bash
git clone https://github.com/abelbg/CI-MLP-Optimizers
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. In PyCharm, mark `src` folder as Source.

## Usage
To run the benchmarking process execute the following command within the `src` folder:

```bash
python main.py
```

## Key Files and Directories
* load_synthetic_data.py: Generates synthetic datasets with configurable noise and complexity.
* mlp.py: Defines the MLP model architecture with ReLU activations and methods to set weights.
* evolution_strategies.py, genetic_algorithm.py, stochastic_gradient_descent.py: Implementations of respective optimization algorithms.
* visualization.py: Utilities for generating exploratory data analysis (EDA) and performance plots.
* main.py: The main driver script that orchestrates the benchmarking and analysis process.

## Results
Benchmark results are stored in the results/ and plots/ directory and include CSV files and plots for performance metrics.