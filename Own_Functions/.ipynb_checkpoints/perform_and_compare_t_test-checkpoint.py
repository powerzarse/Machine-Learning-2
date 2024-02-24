from scipy.stats import ttest_ind

def perform_and_compare_t_test(fold_performances_model1, fold_performances_model2, alpha=0.05):
    """
    Führt einen t-Test zwischen den Leistungen zweier Modelle durch.
    
    Args:
    fold_performances_model1 (list): Liste der Leistungsmetriken für Modell 1.
    fold_performances_model2 (list): Liste der Leistungsmetriken für Modell 2.
    alpha (float): Signifikanzniveau für den t-Test.
    
    Returns:
    None
    """
    
    # Berechnen der durchschnittlichen Performances
    avg_performance_model1 = sum(fold_performances_model1) / len(fold_performances_model1)
    avg_performance_model2 = sum(fold_performances_model2) / len(fold_performances_model2)
    
    # Durchführen des t-Tests
    t_stat, p_value = ttest_ind(fold_performances_model1, fold_performances_model2)
    
    print(f"t-Statistik: {t_stat}, p-Wert: {p_value}")
    if p_value < alpha:
        print("Es gibt einen signifikanten Unterschied zwischen den beiden Modellen.")
        if avg_performance_model1 > avg_performance_model2:
            print("Modell 1 ist statistisch signifikant besser als Modell 2.")
        else:
            print("Modell 2 ist statistisch signifikant besser als Modell 1.")
    else:
        print("Es gibt keinen signifikanten Unterschied zwischen den beiden Modellen.")