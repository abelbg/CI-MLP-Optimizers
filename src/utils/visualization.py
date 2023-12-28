import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime


def plot_eda(X, Y, n_features, noise_level, complexity_level, function_type):
    plt.figure(figsize=(6, 4))
    if n_features == 1:
        plt.scatter(X, Y, alpha=0.6)
        plt.xlabel('Feature')
        plt.ylabel('Target')
    else:
        for feature_index in range(n_features):
            plt.scatter(X[:, feature_index], Y, alpha=0.6, label=f'Feature {feature_index + 1}')
    plt.title(f'{function_type.capitalize()} Function\nNoise: {noise_level}, Complexity: {complexity_level}')
    plt.grid(True)

    file_name = f"plot_{function_type}_features_{n_features}.png"
    plt.savefig(f"../plots/results/data_by_condition/{file_name}")
    plt.close()  # Close the figure after saving


def apply_pca(X, n_components=2):
    X_standardized = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_standardized)
    return principal_components


def plot_pca(X, Y, n_features, noise_level, complexity_level, function_type):
    pca_components = apply_pca(X, n_components=min(n_features, 3))
    plt.figure(figsize=(6, 4))
    if n_features > 2:
        ax = plt.axes(projection='3d')
        ax.scatter(pca_components[:, 0], pca_components[:, 1], pca_components[:, 2], c=Y, cmap='viridis')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
    else:
        plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Y, cmap='viridis')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
    plt.title(f'{function_type.capitalize()} Function\nNoise: {noise_level}, Complexity: {complexity_level}')
    plt.grid(True)
    file_name = f"plot_{function_type}_features_{n_features}_PCA.png"
    plt.savefig(f"../plots/results/data_by_condition/{file_name}")
    plt.close()  # Close the figure after saving


def plot_metrics(df, x_var, metric):
    plt.figure(figsize=(15, 8))
    sns.barplot(x=x_var, y=metric, hue='Optimizer', data=df)
    plt.title(f'{metric} by {x_var} and Optimizer')
    plt.legend(title='Optimizer')
    file_name = f"{metric}_by_{x_var}.png"
    plt.savefig(f'../plots/results/metrics/{file_name}')
    plt.close()


def plot_performance_over_time(data, title):
    plt.figure(figsize=(10, 5))
    for optimizer, values in data.items():
        plt.plot(values, label=optimizer)
    plt.xlabel('Epochs/Generations')
    plt.ylabel('Metric (e.g., Loss)')
    plt.title(title)
    plt.legend()
    # file_name = f"{data}_by_{metric}.png"
    file_name = f"Test_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    plt.savefig(f'../plots/results/performance/{file_name}')
    plt.close()
