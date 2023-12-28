# evolution_strategies.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools


class ESOptimizer:
    def __init__(self, model, input_size, max_hidden_size,
                 population_size=100, num_generations=100,
                 sigma=0.3, epochs=100, batch_size=32):
        self.model = model
        self.input_size = input_size
        self.max_hidden_size = max_hidden_size
        self.population_size = population_size
        self.num_generations = num_generations
        self.sigma = sigma  # Standard deviation for Gaussian noise in mutation
        self.epochs = epochs
        self.batch_size = batch_size

        self.total_weights_biases = (input_size + 1) * max_hidden_size + (max_hidden_size + 1)  # total params count

        # Create DEAP toolbox
        self.toolbox = base.Toolbox()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("select", tools.selBest, k=self.population_size // 2)

    def create_individual(self):
        weights_part = [np.random.uniform(-1, 1) for _ in range(self.total_weights_biases)]
        neurons_part = [np.random.randint(1, self.max_hidden_size + 1)]
        return creator.Individual(weights_part + neurons_part)

    def mutate_individual(self, individual):
        for i in range(self.total_weights_biases):
            if random.random() < self.sigma:
                individual[i] += np.random.normal(0, self.sigma)
        if random.random() < self.sigma:
            individual[self.total_weights_biases] = np.random.randint(1, self.max_hidden_size + 1)
        return individual,

    def evaluate_fitness(self, individual, val_data):
        self.decode_individual(individual)
        val_loss = self.evaluate_model(val_data)
        return (1 / (1 + val_loss),)

    def decode_individual(self, individual, return_weights=False):
        hidden_size = individual[self.total_weights_biases]
        self.model.hidden = nn.Linear(self.input_size, hidden_size)
        self.model.output = nn.Linear(hidden_size, 1)
        weights = individual[:self.total_weights_biases]
        self.model.set_weights(weights)
        if return_weights:
            return weights

    def evaluate_model(self, val_data):
        self.model.eval()
        criterion = nn.MSELoss()
        val_loss = 0

        X_val, Y_val = val_data
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                val_loss += criterion(outputs, targets).item()
        return val_loss / len(val_loader)

    def optimize(self, train_data, val_data):
        population = self.toolbox.population(n=self.population_size)

        for gen in range(self.num_generations):
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, val_data), population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            parents = self.toolbox.select(population)
            offspring = [self.toolbox.clone(ind) for ind in parents]

            for mutant in offspring:
                if random.random() < self.sigma:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, val_data), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        best_weights = self.decode_individual(best_ind, return_weights=True)
        return best_ind, best_weights


if __name__ == "__main__":
    from models.mlp import MLP
    from data.load_synthetic_data import generate_synthetic_data, split_data

    # Data Generation Parameters
    n_samples = 1000
    n_features = 3
    noise_level = 0.1
    complexity_level = 'low'
    function_type = 'sinusoidal'
    train_size = 0.7

    # MLP & SGD Parameters
    hidden_size = 16

    # ES Parameters
    population_size = 4
    num_generations = 3
    sigma = 0.1  # Standard deviation for Gaussian noise in mutation
    epochs = 100
    max_hidden_size = 32
    batch_size = 32

    # Generate and split the synthetic data
    X, Y = generate_synthetic_data(n_samples=n_samples,
                                   n_features=n_features,
                                   noise_level=noise_level,
                                   complexity_level=complexity_level,
                                   function_type=function_type)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, train_size=train_size)

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)

    mlp_model = MLP(input_size=n_features, hidden_size=hidden_size)
    es_optimizer = ESOptimizer(mlp_model,
                               population_size=population_size,
                               num_generations=num_generations,
                               sigma=sigma,
                               epochs=epochs,
                               batch_size=batch_size,
                               input_size=n_features,
                               max_hidden_size=max_hidden_size,)
    print(es_optimizer)
    best_individual = es_optimizer.optimize(train_data, val_data)
    print("Best Individual using ES:", best_individual)