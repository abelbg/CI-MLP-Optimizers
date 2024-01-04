# stochastic_gradient_descent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SGD_Optimizer:
    def __init__(self, model, learning_rate=0.01, epochs=100, batch_size=32, weight_decay=0.01):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_losses = []
        
    def optimize(self, train_data, val_data):
        X_train, Y_train = train_data
        X_val, Y_val = val_data

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(Y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            val_loss = self.evaluate_model(val_data)
            self.val_losses.append(val_loss)

        return self.model

    def evaluate_model(self, val_data):
        self.model.eval()
        criterion = nn.MSELoss()

        X_val, Y_val = val_data
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                val_loss += criterion(outputs, targets).item()
        return val_loss / len(val_loader)


if __name__ == "__main__":
    from models.mlp import MLP
    from data.load_synthetic_data import generate_synthetic_data, split_data

    # Data Generation Parameters
    n_samples = 1000
    n_features = 10
    noise_level = 0.1
    complexity_level = 'low'
    function_type = 'sinusoidal'
    train_size = 0.7
    # MLP & SGD Parameters
    hidden_size = 16
    learning_rate = 0.01
    epochs = 100
    batch_size = 32

    # Generate and split the data
    X, Y = generate_synthetic_data(n_samples=n_samples,
                                   n_features=n_features,
                                   noise_level=noise_level,
                                   complexity_level=complexity_level,
                                   function_type=function_type)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, train_size)

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)

    mlp_model = MLP(input_size=n_features, hidden_size=hidden_size)
    sgd_optimizer = SGD_Optimizer(mlp_model, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)

    best_model = sgd_optimizer.optimize(train_data, val_data)

    val_loss = sgd_optimizer.evaluate_model(val_data)
    print(f'Validation Loss: {val_loss:.4f}')
