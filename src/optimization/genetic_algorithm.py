# genetic_algorithm.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools
import numpy as np
import random


class GAOptimizer:
    def __init__(self, model, input_size, max_hidden_size,
                 population_size=50, num_generations=100,
                 crossover_prob=0.8, mutation_prob=0.05,
                 epochs=100, batch_size=32, sigma=0.1):
        self.model = model
        self.input_size = input_size
        self.max_hidden_size = max_hidden_size
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.epochs = epochs
        self.batch_size = batch_size
        self.sigma = sigma  # Standard deviation for Gaussian mutation

        self.total_weights_biases = self.calculate_total_params(input_size, max_hidden_size)

        # Create the DEAP toolbox
        self.toolbox = base.Toolbox()

        # Define the fitness function
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Define the individual and population
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Define the genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_fitness)

    def calculate_total_params(self, input_size, max_hidden_size):
        # Calculate the total number of weights and biases for the maximum configuration
        return (input_size + 1) * max_hidden_size + (max_hidden_size + 1)

    def create_individual(self):
        # Create a part for weights and biases
        weights_part = [np.random.uniform(-1, 1) for _ in range(self.total_weights_biases)]
        # Add a gene for the number of neurons (1 to 32)
        neurons_part = [np.random.randint(1, self.max_hidden_size + 1)]

        return creator.Individual(weights_part + neurons_part)

    def mutate(self, individual):
        # Mutate weights and biases with Gaussian noise
        for i in range(self.total_weights_biases):
            if random.random() < self.mutation_prob:
                individual[i] += np.random.normal(0, self.sigma)
        # Mutate number of neurons
        if random.random() < self.mutation_prob:
            individual[self.total_weights_biases] = np.random.randint(1, self.max_hidden_size + 1)

        return individual,

    def evaluate_fitness(self, individual, train_data, val_data):
        # Decode the individual to set the model's parameters
        self.decode_individual(individual)
        # Evaluate the model on validation data
        val_loss = self.evaluate_model(val_data)
        # The fitness could be the inverse of validation loss (as we aim to minimize the loss)
        return (1 / (1 + val_loss),)

    def decode_individual(self, individual, return_weights=False):
        hidden_size = int(individual[self.total_weights_biases])
        self.model.set_hidden_layer(hidden_size)
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

        # train_data = list(zip(train_data[0], train_data[1]))
        # val_data = list(zip(val_data[0], val_data[1]))

        population = self.toolbox.population(n=self.population_size)

        # Evaluate the entire population
        for gen in range(self.num_generations):
            # Evaluate, select, mate, and mutate
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, train_data, val_data), population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, train_data, val_data), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the population
            population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        best_weights = self.decode_individual(best_ind, return_weights=True)
        return best_ind, best_weights


if __name__ == "__main__":
    from models.mlp import MLP
    from data.load_synthetic_data import generate_synthetic_data, split_data

    # Data Generation Parameters
    n_samples = 1000
    n_features = 10
    noise_level = 0.1
    complexity_level = 'high'
    function_type = 'sinusoidal'
    train_size = 0.7

    # MLP & SGD Parameters
    hidden_size = 16

    # GA Parameters
    population_size = 4
    num_generations = 3
    crossover_prob = 0.8
    mutation_prob = 0.1
    epochs = 100
    max_hidden_size = 32
    batch_size = 32

    X, Y = generate_synthetic_data(n_samples=n_samples,
                                   n_features=n_features,
                                   noise_level=noise_level,
                                   complexity_level=complexity_level,
                                   function_type=function_type)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, train_size=train_size)

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)

    mlp_model = MLP(input_size=n_features, hidden_size=hidden_size)
    ga_optimizer = GAOptimizer(mlp_model,
                               population_size=population_size,
                               num_generations=num_generations,
                               crossover_prob=crossover_prob,
                               mutation_prob=mutation_prob,
                               epochs=epochs,
                               input_size=n_features,
                               max_hidden_size=max_hidden_size,
                               batch_size=batch_size)
    print(ga_optimizer)
    best_individual = ga_optimizer.optimize(train_data, val_data)
    print("Best Individual:", best_individual)

