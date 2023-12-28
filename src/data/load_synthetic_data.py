# load_synthethic_data.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples, noise_level, complexity_level, n_features=1, function_type='linear', noise_scaling_factor=0.05):
    
    complexity_mapping = {'low': 1, 'medium': 2, 'high': 3}
    complexity_level = complexity_mapping[complexity_level]
    
    np.random.seed(0)  # for reproducibility
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    Y = np.zeros((n_samples, ))

    # Generate the polynomial or sinusoidal features
    if function_type == 'polynomial':
        for i in range(n_features):
            for power in range(1, complexity_level + 1):
                Y += (np.random.rand() * X[:, i] ** power)
    elif function_type == 'sinusoidal':
        for i in range(n_features):
            Y += np.sin(complexity_level * X[:, i])
    else:
        raise ValueError("Unsupported function type provided.")

    # Adjust noise based on the range of Y values
    Y_range = np.ptp(Y)  # Peak to peak (max - min) of Y values
    adjusted_noise = np.random.normal(0, noise_level * Y_range * noise_scaling_factor, n_samples)
    Y += adjusted_noise

    # Normalization is optional based on whether you want it or not
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    return X, Y


def split_data(X, Y, train_size):
    X_train, X_remaining, Y_train, Y_remaining = train_test_split(X, Y, train_size=train_size)
    X_val, X_test, Y_val, Y_test = train_test_split(X_remaining, Y_remaining, test_size=0.5)

    # Reshape the target columns
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def plot_data(X, Y, title):
    if X.shape[1] == 1:
        plt.scatter(X, Y, alpha=0.6)
    else:
        for i in range(X.shape[1]):
            plt.scatter(X[:, i], Y, alpha=0.6, label=f'Feature {i+1}')
        plt.legend()

    plt.title(title)
    plt.xlabel('Input Features')
    plt.ylabel('Target Output')
    plt.grid(True)
    plt.show()

    
if __name__ == "__main__":
    n_samples_train = 1000
    n_samples_val_test = 1000
    noise_level = 0.1
    complexity_level = 1
    n_features = 2
    function_types = ['polynomial', 'sinusoidal']

    for f_type in function_types:
        X_train, Y_train = generate_synthetic_data(n_samples_train, noise_level, complexity_level,
                                                   n_features=n_features, function_type=f_type)
        X_val_test, Y_val_test = generate_synthetic_data(n_samples_val_test, noise_level, complexity_level,
                                                         n_features=n_features, function_type=f_type)

        # Split the data using only train_size
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X_val_test, Y_val_test, train_size=0.7)

        plot_data(X_train, Y_train, f'Training Data - {f_type}')
        plot_data(X_val, Y_val, f'Validation Data - {f_type}')
        plot_data(X_test, Y_test, f'Test Data - {f_type}')
