import matplotlib.pyplot as plt

def plot_comparison_across_folds(fold_histories, metric1, metric2, title):
    """
    Function to plot a comparison of two metrics across folds.

    Args:
    fold_histories (list): List of dictionaries containing the metrics across folds.
    metric1 (str): The first metric to plot.
    metric2 (str): The second metric to plot.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot for the first metric (loss)
    plt.subplot(1, 2, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history[metric1], label=f'Fold {i+1} {metric1}')
        plt.plot(history[metric2], label=f'Fold {i+1} {metric2}', linestyle='--')
    plt.title(f'{metric1} vs {metric2}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # Adjusting layout
    plt.tight_layout()
    
    plt.suptitle(title)
    plt.show()
