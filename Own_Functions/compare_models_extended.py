import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score

def compare_models(model_A, model_B, test_generator, model_names, alpha=0.05):
    """
    Vergleicht zwei Modelle anhand ihrer Vorhersagen auf einem Testdatensatz.

    Args:
    model_A: Das erste zu vergleichende Modell.
    model_B: Das zweite zu vergleichende Modell.
    test_generator: Ein Keras-Generator, der die Testdaten liefert.
    model_names: Ein Dictionary, das Modellobjekte auf ihre Namen abbildet.
    alpha: Das Signifikanzniveau für den t-Test, standardmäßig 0.05.

    Returns:
    Eine Tuple, bestehend aus der Genauigkeit von Modell A, der Genauigkeit von Modell B,
    dem p-Wert des t-Tests und einer Zeichenkette, die angibt, welches Modell besser ist oder 'None' für keinen signifikanten Unterschied.
    """

    predictions_A = []
    predictions_B = []
    true_labels = []

    # Gehe durch den Testgenerator und sammle Vorhersagen und wahre Labels
    for i in range(len(test_generator)):
        X, y = test_generator[i]
        true_labels.extend(y)

        preds_A = model_A.predict(X, verbose=0)
        preds_B = model_B.predict(X, verbose=0)

        predictions_A.extend(preds_A)
        predictions_B.extend(preds_B)

    # Konvertiere Listen in NumPy Arrays für die weitere Analyse
    predictions_A = np.array(predictions_A)
    predictions_B = np.array(predictions_B)
    true_labels = np.array(true_labels)

    # Finde die vorhergesagten Labels als Indizes mit der höchsten Wahrscheinlichkeit
    predicted_labels_A = np.argmax(predictions_A, axis=1)
    predicted_labels_B = np.argmax(predictions_B, axis=1)
    true_labels = np.argmax(true_labels, axis=1)

    # Berechne die Genauigkeit für jedes Modell
    accuracy_A = accuracy_score(true_labels, predicted_labels_A)
    accuracy_B = accuracy_score(true_labels, predicted_labels_B)

    # Führe den gepaarten t-Test auf den vorhergesagten Labels durch
    t_statistic, p_value = ttest_rel(predicted_labels_A, predicted_labels_B)

    # Hole die Namen der Modelle aus dem model_names Dictionary
    model_A_name = model_names.get(model_A, 'Unknown Model A')
    model_B_name = model_names.get(model_B, 'Unknown Model B')

    # Drucke die Ergebnisse des Vergleichs
    print("\nComparison between", model_A_name, "and", model_B_name)
    print(f"Accuracy {model_A_name}: {accuracy_A:.4f}")
    print(f"Accuracy {model_B_name}: {accuracy_B:.4f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Interpretiere das Ergebnis des t-Tests
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference.")
        if accuracy_A > accuracy_B:
            print(f"{model_A_name} is statistically significantly better than {model_B_name}.")
        else:
            print(f"{model_B_name} is statistically significantly better than {model_A_name}.")
    else:
        print("Fail to reject the null hypothesis. No significant difference.")
        if accuracy_A == accuracy_B:
            print(f"Both {model_A_name} and {model_B_name} have similar performance.")
        elif accuracy_A > accuracy_B:
            print(f"{model_A_name} is better, but not significantly.")
        else:
            print(f"{model_B_name} is better, but not significantly.")

    # Gib die berechneten Werte zurück
    return accuracy_A, accuracy_B, p_value, "A" if accuracy_A > accuracy_B else "B" if accuracy_B > accuracy_A else "None"
