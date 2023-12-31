# main.py
import time
import json
import logging
import pandas as pd
import torch
from models.mlp import MLP
from optimization.stochastic_gradient_descent import SGD_Optimizer
from optimization.genetic_algorithm import GAOptimizer
from optimization.evolution_strategies import ESOptimizer
from data.load_synthetic_data import generate_synthetic_data, split_data
from visualization.plots import plot_eda, plot_pca, plot_metrics, plot_metric_distribution

# Set a fixed seed for reproducibility
torch.manual_seed(0)

# Define paths to CSV files
BENCHMARK_RESULTS_PATH = '../results/benchmark_results.csv'
AGGREGATED_RESULTS_PATH = '../results/aggregated_results.csv'


def benchmark_optimizer(optimizer, train_data, val_data, test_data, optimizer_name,
                        noise_level, n_features, function_type, complexity_level):
    start_time = time.time()

    if optimizer_name in ['GA', 'ES']:
        _, hidden_neurons = optimizer.optimize(train_data, val_data)
    else:
        optimizer.optimize(train_data, val_data)
        hidden_neurons = optimizer.model.hidden.out_features

    training_time = time.time() - start_time

    train_loss = optimizer.evaluate_model(train_data)
    val_loss = optimizer.evaluate_model(val_data)
    test_loss = optimizer.evaluate_model(test_data)

    results = {
        'Optimizer': optimizer_name,
        'Feature_Dimension': n_features,
        'Complexity_Level': complexity_level,
        'Noise_Level': noise_level,
        'Hidden_Neurons': hidden_neurons,
        'Function_Type': function_type,
        'Training_Loss': train_loss,
        'Validation_Loss': val_loss,
        'Test_Loss': test_loss,
        'Training_Time': training_time,
    }
    return results


def aggregate_results(benchmark_results):
    df_results = pd.DataFrame(benchmark_results)
    df_agg_results = df_results.groupby(['Optimizer',
                                         'Feature_Dimension',
                                         'Complexity_Level',
                                         'Noise_Level',
                                         'Hidden_Neurons']
                                        ).mean().reset_index()
    return df_results, df_agg_results


def get_best_by_optimizer(df, metric='Validation_Loss'):
    best_results = {}
    for optimizer in df['Optimizer'].unique():
        best_row = df[df['Optimizer'] == optimizer].sort_values(by=metric).iloc[0]
        best_results[optimizer] = best_row
    return best_results


def get_overall_best(df, metric='Validation_Loss'):
    return df.sort_values(by=metric).iloc[0]


def reconstruct_network(entry):
    input_size = entry['Feature_Dimension']
    hidden_size = entry['Hidden_Neurons']
    mlp_model = MLP(input_size=input_size, hidden_size=hidden_size)

    return mlp_model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(),]
    )

    logging.info("Benchmarking process started.")

    # Define the conditions for benchmarking
    conditions = [
        {'n_features': dim, 'noise_level': noise, 'complexity_level': complexity}
        for dim in [3, 6, 20]  # Low, Medium dimensionality
        for noise in [0.1, 0.5, 1.0]  # Low, High noise levels
        for complexity in ['low', 'medium', 'high']
    ]

    total_conditions = len(conditions)
    processed_conditions = 0

    log_best_results = {
        'by_condition': {},
        'by_algorithm': {optimizer: None for optimizer in ['SGD', 'GA', 'ES']}
    }

    benchmark_results = []
    for condition in conditions:
        processed_conditions += 1
        remaining_conditions = total_conditions - processed_conditions
        logging.info(f"Processing condition {processed_conditions}/{total_conditions}: {condition}")

        condition_start_time = time.time()

        # Generate data
        X, Y = generate_synthetic_data(n_samples=3000, noise_level=condition['noise_level'],
                                       complexity_level=condition['complexity_level'],
                                       n_features=condition['n_features'], function_type='polynomial')
        train_data, val_data, test_data = split_data(X, Y, train_size=0.7)

        plot_eda(X, Y, condition['n_features'], condition['noise_level'],
                 condition['complexity_level'])
        plot_pca(X, Y, condition['n_features'], condition['noise_level'],
                 condition['complexity_level'])

        mlp_model = MLP(input_size=condition['n_features'], hidden_size=16)

        optimizers = {
            'SGD': SGD_Optimizer(mlp_model),
            'GA': GAOptimizer(mlp_model, input_size=condition['n_features']),
            'ES': ESOptimizer(mlp_model, input_size=condition['n_features'])
        }

        for optimizer_name, optimizer in optimizers.items():
            logging.info(f"Optimizing with {optimizer_name}")
            result = benchmark_optimizer(optimizer, train_data, val_data, test_data, optimizer_name,
                                         condition['noise_level'], condition['n_features'],
                                         'polynomial', condition['complexity_level'])
            benchmark_results.append(result)
            logging.info(f"Completed optimization with {optimizer_name}. Result: {result}")

            # Update best results in logging
            current_best = log_best_results['by_algorithm'][optimizer_name]
            if current_best is None or current_best['Validation_Loss'] > result['Validation_Loss']:
                log_best_results['by_algorithm'][optimizer_name] = result

            condition_key = f"{condition['noise_level']}_{condition['n_features']}_polynomial"
            current_condition_best = log_best_results['by_condition'].get(condition_key)
            if current_condition_best is None or current_condition_best['Validation_Loss'] > result['Validation_Loss']:
                log_best_results['by_condition'][condition_key] = result

        condition_end_time = time.time()
        condition_duration = condition_end_time - condition_start_time
        logging.info(f"Total execution time for condition: {condition_duration:.2f} seconds")
        logging.info(f"Remaining conditions: {remaining_conditions}")

    # Get results
    df_results, df_agg_results = aggregate_results(benchmark_results)

    # Plotting results
    for metric in ['Training_Loss', 'Validation_Loss', 'Test_Loss', 'Training_Time']:
        for x_var in ['Feature_Dimension', 'Noise_Level', 'Complexity_Level']:
            plot_metrics(df_agg_results, x_var, metric)

    df_results.to_csv(BENCHMARK_RESULTS_PATH, index=False)
    df_agg_results.to_csv(AGGREGATED_RESULTS_PATH, index=False)

    logging.info("\nBest Results by Condition:")
    for condition, result in log_best_results['by_condition'].items():
        logging.info(f"Condition {condition}: {json.dumps(result, indent=2)}")
        network = reconstruct_network(result)
        logging.info("Reconstructed Network for the best condition:")
        logging.info(network)

    logging.info("\nBest Results by Algorithm:")
    for algorithm, result in log_best_results['by_algorithm'].items():
        logging.info(f"Algorithm {algorithm}: {json.dumps(result, indent=2)}")
        network = reconstruct_network(result)
        logging.info("Reconstructed Network for the best algorithm:")
        logging.info(network)

    with open('../results/best_results_by_condition.json', 'w') as file:
        json.dump(log_best_results['by_condition'], file, indent=2)

    with open('../results/best_results_by_algorithm.json', 'w') as file:
        json.dump(log_best_results['by_algorithm'], file, indent=2)

    logging.info("Benchmarking process completed.")


if __name__ == "__main__":
    main()