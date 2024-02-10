import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score

def compare_models(model_A, model_B, test_data, validation_generator, alpha=0.05):
    # Assuming you have true labels for the test set
    true_labels = validation_generator.classes  # Ground truth labels

    # Make predictions on the test set for both models
    predicted_labels_model_A = model_A.predict(test_data)
    predicted_labels_model_B = model_B.predict(test_data)

    # Convert predicted probabilities to class labels
    predicted_labels_model_A = np.argmax(predicted_labels_model_A, axis=1)
    predicted_labels_model_B = np.argmax(predicted_labels_model_B, axis=1)

    # Calculate the test statistic (accuracy)
    accuracy_model_A = accuracy_score(true_labels, predicted_labels_model_A)
    accuracy_model_B = accuracy_score(true_labels, predicted_labels_model_B)

    # Perform a paired t-test
    t_statistic, p_value = ttest_rel(accuracy_model_A, accuracy_model_B)

    # Check if the p-value is less than alpha
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis. No significant difference.")

    # Print additional information
    print(f"Accuracy Model A: {accuracy_model_A:.4f}")
    print(f"Accuracy Model B: {accuracy_model_B:.4f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

## Example usage:
#compare_models(your_model_A, your_model_B, your_test_data, your_validation_generator)
