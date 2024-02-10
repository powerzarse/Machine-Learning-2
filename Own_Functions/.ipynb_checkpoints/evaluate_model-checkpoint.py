import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score

def evaluate_model(model, validation_generator, alpha=0.05):
    """
    Test if the model's performance is significantly different from random chance.

    Args:
    model: The trained model to evaluate.
    validation_generator: Generator yielding validation data.
    alpha (float): The significance level for the test.

    Returns:
    tuple: A tuple containing the observed accuracy, t-statistic, and p-value.
    """
    # Assuming you have true labels for the validation set
    true_labels = validation_generator.classes  # Assuming you're using flow_from_directory

    # Make predictions on the validation set
    predicted_labels = model.predict(validation_generator)

    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(predicted_labels, axis=1)

    # Calculate the test statistic (accuracy)
    observed_accuracy = accuracy_score(true_labels, predicted_labels)

    # Perform a paired t-test
    t_statistic, p_value = ttest_rel(true_labels, predicted_labels)

    # Check if the p-value is less than alpha
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis. No significant difference.")

    # Print additional information
    print(f"Observed Accuracy: {observed_accuracy:.4f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    return observed_accuracy, t_statistic, p_value

# Example usage:
# observed_accuracy, t_statistic, p_value = evaluate_model(your_trained_model, your_validation_generator)
