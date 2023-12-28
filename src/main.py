# main.py
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models.mlp import MLP
from optimization.stochastic_gradient_descent import SGD_Optimizer
from optimization.genetic_algorithm import GAOptimizer
from optimization.evolution_strategies import ESOptimizer
from data.load_synthetic_data import generate_synthetic_data, split_data

# Set a fixed seed for reproducibility
torch.manual_seed(0)


def benchmark_optimizer(optimizer, train_data, val_data, test_data, optimizer_name, noise_level, n_features,
                        function_type):
    start_time = time.time()
    optimizer.optimize(train_data, val_data)
    training_time = time.time() - start_time

    train_loss = optimizer.evaluate_model(train_data)
    val_loss = optimizer.evaluate_model(val_data)
    test_loss = optimizer.evaluate_model(test_data)

    results = {
        'Optimizer': optimizer_name,
        'Training_Loss': train_loss,
        'Validation_Loss': val_loss,
        'Test_Loss': test_loss,
        'Training_Time': training_time,
        'Hidden_Neurons': optimizer.model.hidden.out_features,
        'Noise_Level': noise_level,
        'Feature_Dimension': n_features,
        'Function_Type': function_type
    }

    return results


def generate_plot(df, x_var, metric):
    plt.figure(figsize=(15, 8))
    sns.barplot(x=x_var, y=metric, hue='Optimizer', data=df)
    plt.title(f'{metric} by {x_var} and Optimizer')
    plt.legend(title='Optimizer')
    plt.savefig(f'../plots/results/{metric}_by_{x_var}_and_optimizer.png')
    plt.show()


def get_best_by_optimizer(df, metric='Validation_Loss'):
    best_results = {}
    for optimizer in df['Optimizer'].unique():
        best_row = df[df['Optimizer'] == optimizer].sort_values(by=metric).iloc[0]
        best_results[optimizer] = best_row
    return best_results


def get_overall_best(df, metric='Validation_Loss'):
    return df.sort_values(by=metric).iloc[0]


def reconstruct_network(entry, optimizer_name):
    input_size = entry['Feature_Dimension']

    if optimizer_name in ['GA', 'ES']:
        hidden_size = entry.get('Evolved_Hidden_Size')
        mlp_model = MLP(input_size=input_size, hidden_size=hidden_size)
        evolved_weights = entry.get('Evolved_Weights')
        if evolved_weights:
            mlp_model.set_weights(evolved_weights)
    else:
        hidden_size = entry['Hidden_Neurons']
        mlp_model = MLP(input_size=input_size, hidden_size=hidden_size)
    return mlp_model


def main():
    # Define the conditions for benchmarking
    conditions = [
        {'n_features': dim, 'function_type': func_type, 'noise_level': noise, 'complexity_level': complexity}
        for noise in [0.05, 0.1]  # Low, Medium, High noise levels
        for dim in [2, 5]          # Low, Medium, High dimensionality
        for func_type in ['sinusoidal']
        for complexity in ['low', 'medium']
    ]

    # Initialize a dictionary for logging best results
    log_best_results = {
        'by_condition': {},
        'by_algorithm': {optimizer: None for optimizer in ['SGD', 'GA', 'ES']}
    }

    benchmark_results = []
    for condition in conditions:
        X, Y = generate_synthetic_data(n_samples=1000, noise_level=condition['noise_level'],
                                       complexity_level=condition['complexity_level'],
                                       n_features=condition['n_features'], function_type=condition['function_type'])
        train_data, val_data, test_data = split_data(X, Y, train_size=0.7)

        mlp_model = MLP(input_size=condition['n_features'], hidden_size=16)

        # Benchmark each optimizer with specific parameters
        optimizers = {
            'SGD': SGD_Optimizer(mlp_model, learning_rate=0.01, epochs=100, batch_size=32),
            'GA': GAOptimizer(mlp_model, input_size=condition['n_features'], max_hidden_size=32,
                              population_size=50, num_generations=100, crossover_prob=0.8, mutation_prob=0.05,
                              epochs=100, batch_size=32, sigma=0.1),
            'ES': ESOptimizer(mlp_model, input_size=condition['n_features'], max_hidden_size=32,
                              population_size=100, num_generations=100, sigma=0.3,
                              epochs=100, batch_size=32)
        }

        for optimizer_name, optimizer in optimizers.items():
            result = benchmark_optimizer(optimizer, train_data, val_data, test_data, optimizer_name,
                                         condition['noise_level'], condition['n_features'], condition['function_type'])
            benchmark_results.append(result)

            # Update best results in logging
            current_best = log_best_results['by_algorithm'][optimizer_name]
            if current_best is None or current_best['Validation_Loss'] > result['Validation_Loss']:
                log_best_results['by_algorithm'][optimizer_name] = result

            condition_key = f"{condition['noise_level']}_{condition['n_features']}_{condition['function_type']}"
            current_condition_best = log_best_results['by_condition'].get(condition_key)
            if current_condition_best is None or current_condition_best['Validation_Loss'] > result['Validation_Loss']:
                log_best_results['by_condition'][condition_key] = result

    # Aggregating and plotting results
    df_results = pd.DataFrame(benchmark_results)
    df_agg_results = df_results.groupby(['Optimizer', 'Hidden_Neurons',
                                         'Noise_Level', 'Feature_Dimension', 'Function_Type']).mean().reset_index()

    for metric in ['Training_Loss', 'Validation_Loss', 'Test_Loss', 'Training_Time']:
        for x_var in ['Feature_Dimension', 'Noise_Level', 'Function_Type']:
            generate_plot(df_agg_results, x_var, metric)

    # Save results to CSV
    df_results.to_csv('../results/benchmark_results.csv', index=False)
    df_agg_results.to_csv('../results/aggregated_results.csv', index=False)

    # Display and save logged information
    print("\nBest Results by Condition:")
    for condition, result in log_best_results['by_condition'].items():
        print(f"Condition {condition}: {json.dumps(result, indent=2)}")
        # Reconstruction of the network for the best condition
        network = reconstruct_network(result, result['Optimizer'])
        print("Reconstructed Network for the best condition:")
        print(network)

    print("\nBest Results by Algorithm:")
    for algorithm, result in log_best_results['by_algorithm'].items():
        print(f"Algorithm {algorithm}: {json.dumps(result, indent=2)}")
        # Reconstruction of the network for the best algorithm
        network = reconstruct_network(result, algorithm)
        print("Reconstructed Network for the best algorithm:")
        print(network)

    # Save logged information to JSON files
    with open('../results/best_results_by_condition.json', 'w') as file:
        json.dump(log_best_results['by_condition'], file, indent=2)

    with open('../results/best_results_by_algorithm.json', 'w') as file:
        json.dump(log_best_results['by_algorithm'], file, indent=2)


if __name__ == "__main__":
    main()
