import matplotlib.pyplot as plt

def plot_metric_across_folds(fold_histories, metric, title):
    """
    Function to plot a metric across folds.

    Args:
    fold_histories (list): List of dictionaries containing the metrics across folds.
    metric (str): The metric to plot.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(4, 4))
    for i, history in enumerate(fold_histories):
        plt.plot(history[metric], label=f'Fold {i+1}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.show()
