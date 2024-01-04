# evolution_strategies.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools


# evolution_strategies.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools

class ESOptimizer:
    def __init__(self, model, input_size, max_hidden_size=32,
                 population_size=200, num_generations=50,
                 sigma=0.3, batch_size=32):
        self.model = model
        self.input_size = input_size
        self.max_hidden_size = max_hidden_size
        self.population_size = population_size
        self.num_generations = num_generations
        self.sigma = sigma  # Standard deviation for Gaussian noise in mutation
        self.batch_size = batch_size
        self.total_weights_biases = (input_size + 1) * max_hidden_size + (max_hidden_size + 1)  # total params count
        self.val_losses = [] 

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
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def create_individual(self):
        weights_part = [np.random.uniform(-1, 1) for _ in range(self.total_weights_biases)]
        neurons_part = [np.random.randint(10, self.max_hidden_size + 1)]
        return creator.Individual(weights_part + neurons_part)

    def mutate_individual(self, individual):
        for i in range(len(individual) - 1):  # Exclude the last gene (hidden layer size)
            if random.random() < self.sigma:
                individual[i] += np.random.normal(0, self.sigma)
        if random.random() < self.sigma:
            individual[-1] = np.random.randint(10, self.max_hidden_size + 1)
        return individual,

    def evaluate_fitness(self, individual, val_data):
        self.decode_individual(individual)
        val_loss = self.evaluate_model(val_data)
        return (1 / (1 + val_loss),)

    def decode_individual(self, individual):
        hidden_size = individual[-1]
        weights = individual[:-1]
        self.model.set_hidden_layer(hidden_size, weights)

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

            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))  # 'k' is the number of individuals to select

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply mutation to the offspring
            for mutant in offspring:
                if random.random() < self.sigma:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, val_data), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The offspring is now the new population
            population[:] = offspring
            best_ind = tools.selBest(population, 1)[0]
            self.val_losses.append(1 / best_ind.fitness.values[0] - 1) # Invert fitness to get loss

        # Select the best individual from the final population
        best_ind = tools.selBest(population, 1)[0]
        hidden_size = best_ind[-1]
        return best_ind, hidden_size


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
    population_size = 100
    num_generations = 10
    sigma = 0.1  # Standard deviation for Gaussian noise in mutation
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
                               batch_size=batch_size,
                               input_size=n_features,
                               max_hidden_size=max_hidden_size,)
    print(es_optimizer)
    best_individual = es_optimizer.optimize(train_data, val_data)
    print("Best Individual using ES:", best_individual)
