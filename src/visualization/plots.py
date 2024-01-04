import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime
import os


def plot_eda(X, Y, n_features, noise_level, complexity_level):
    plt.figure(figsize=(6, 4))
    if n_features == 1:
        plt.scatter(X, Y, alpha=0.6)
        plt.xlabel('Feature')
        plt.ylabel('Target')
    else:
        for feature_index in range(n_features):
            plt.scatter(X[:, feature_index], Y, alpha=0.6, label=f'Feature {feature_index + 1}')
    plt.title(f'Polynomial Function\nNoise: {noise_level}, Complexity: {complexity_level}')
    plt.grid(True)

    file_name = f"plot_polynomial_features_{n_features}.png"
    plt.savefig(f"../plots/results/data_by_condition/{file_name}")
    plt.close()  # Close the figure after saving


def apply_pca(X, n_components=2):
    X_standardized = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_standardized)
    return principal_components


def plot_pca(X, Y, n_features, noise_level, complexity_level):
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
    plt.title(f'Polynomial Function\nNoise: {noise_level}, Complexity: {complexity_level}')
    plt.grid(True)
    file_name = f"plot_polynomial_features_{n_features}_PCA.png"
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
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.title(title)
    plt.legend()
    file_name = f"BestFitness_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    plt.savefig(f'../plots/results/performance/{file_name}')
    plt.close()


def plot_losses(optimizers, condition):
    for name, optimizer in optimizers.items():
        plt.plot(optimizer.val_losses, label=name)
    plt.title(f'Losses for condition: {condition}')
    plt.xlabel('Iteration/Generation')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    condition_str = '_'.join(str(value) for value in condition.values())
    filename = f'losses_{condition_str}.png'
    filepath = os.path.join('../plots/results/performance/', filename)  # replace with your directory
    plt.savefig(filepath)
    plt.close()


def generate_subplots(df, x_var, metrics):
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(15, 8))
    for ax, metric in zip(axes, metrics):
        sns.barplot(x=x_var, y=metric, hue='Optimizer', data=df, ax=ax)
        ax.set_title(f'{metric} by {x_var} and Optimizer')
        ax.legend(title='Optimizer')
    plt.close()


def improve_plot_aesthetics():
    sns.set(style="whitegrid")
    plt.title('Title')
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.close()

def plot_metric_distribution(df, metric):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Optimizer', y=metric, data=df)
    plt.title(f'Distribution of {metric} for each optimizer')
    plt.show()