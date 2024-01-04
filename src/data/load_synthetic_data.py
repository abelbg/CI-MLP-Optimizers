import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_data(n_samples, noise_level, complexity_level, n_features=1, function_type='linear', noise_scaling_factor=0.05):
    complexity_mapping = {'low': 1, 'medium': 2, 'high': 3}
    complexity_level = complexity_mapping[complexity_level]

    np.random.seed(0)  # for reproducibility
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    Y = np.zeros((n_samples, ))

    # Generate the polynomial or sinusoidal features with interactions
    for i in range(n_features):
        if function_type == 'polynomial':
            for power in range(1, complexity_level + 1):
                Y += (np.random.rand() * X[:, i] ** power)
                # Introduce interactions between features
                if i > 0:
                    Y += (np.random.rand() * X[:, i] * X[:, i - 1])
        elif function_type == 'sinusoidal':
            Y += np.sin(complexity_level * X[:, i])
            # Introduce frequency modulation based on other features
            if i > 0:
                Y += np.sin(complexity_level * X[:, i] * X[:, i - 1])
        else:
            raise ValueError("Unsupported function type provided.")

    # Add non-linear noise
    Y_range = np.ptp(Y)  # Peak to peak (max - min) of Y values
    adjusted_noise = np.random.normal(0, noise_level * Y_range * noise_scaling_factor, n_samples)
    non_linear_noise = adjusted_noise * np.sin(X[:, 0])
    Y += non_linear_noise

    # Structured noise that depends on the magnitude of input features
    structured_noise = np.sum(X, axis=1) * (np.random.rand(n_samples) - 0.5) * noise_level
    Y += structured_noise

    # Normalize features
    scaler_X = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)

    # Normalize target
    Y_normalized = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    return X_normalized, Y_normalized


def split_data(X, Y, train_size):
    X_train, X_remaining, Y_train, Y_remaining = train_test_split(X, Y, train_size=train_size)
    X_val, X_test, Y_val, Y_test = train_test_split(X_remaining, Y_remaining, test_size=0.5)

    # Reshape the target columns
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def plot_data(X, Y, title, ax):

    if X.shape[1] == 1:
        ax.scatter(X, Y, alpha=0.6)
    else:
        for i in range(X.shape[1]):
            ax.scatter(X[:, i], Y, alpha=0.6, label=f'Feature {i+1}')

    ax.set_title(title)
    ax.set_xlabel('Input Features')
    ax.set_ylabel('Target Output')
    ax.grid(True)


if __name__ == "__main__":
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("rocket")  # Set the color palette to 'rocket'
    plt.rcParams.update({'font.size': 14})  # Adjust font size for readability

    configurations = [
        (1000, 0.1, 'low', 2, 'polynomial'),
        (1000, 0.1, 'low', 2, 'sinusoidal'),
        (1000, 0.5, 'medium', 6, 'polynomial'),
        (1000, 0.5, 'medium', 6, 'sinusoidal'),
        (1000, 1.0, 'high', 10, 'polynomial'),
        (1000, 1.0, 'high', 10, 'sinusoidal'),
    ]

    # Create a figure for the composite graph
    n_rows = 3  # or any other layout configuration
    n_cols = 2  # or any other layout configuration
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))  # Adjust the figure size as needed

    for idx, config in enumerate(configurations):
        n_samples, noise_level, complexity_level, n_features, function_type = config
        X, Y = generate_synthetic_data(n_samples, noise_level, complexity_level, n_features, function_type)

        title = f'{function_type.capitalize()} Function with {n_features} Feature(s)\n'
        title += f'Noise: {noise_level}, Complexity: {complexity_level}'

        # Determine the subplot to use
        ax = axs[idx // n_cols, idx % n_cols]

        # Plot and show the synthetic data
        plot_data(X, Y, title, ax)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
