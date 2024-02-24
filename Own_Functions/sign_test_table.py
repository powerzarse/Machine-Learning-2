from scipy.stats import binom

def sign_test_table(model1, results_alg1, model2, results_alg2, alpha=0.05):
    # Paare von Ergebnissen für beide Algorithmen erstellen
    paired_results = list(zip(results_alg1, results_alg2))
    
    # Zähle, wie oft das erste Modell besser ist als das zweite
    wins_alg1 = sum(1 for a1, a2 in paired_results if a1 > a2)
    
    # Zähle, wie oft das zweite Modell besser ist als das erste
    wins_alg2 = sum(1 for a1, a2 in paired_results if a1 < a2)
    
    # Zähle Unentschieden
    ties = sum(1 for a1, a2 in paired_results if a1 == a2)
    
    # Berechne die Gesamtzahl der Fälle ohne Unentschieden
    n = len(paired_results) - ties

    # Wenn alle Vergleiche Unentschieden sind, ist der P-Wert 1
    if n == 0:
        return 1.0, False, None  # Kein eindeutiger Gewinner
    
    # Bestimme die maximale Anzahl von Gewinnen
    T = max(wins_alg1, wins_alg2)
    
    # Berechne den P-Wert
    p_value = 2 * (1 - binom.cdf(T - 1, n, 0.5))
    
    # Stelle sicher, dass der P-Wert nie größer als 1 ist
    p_value = min(p_value, 1.0)
    
    # Entscheide, ob die Nullhypothese abgelehnt wird
    reject_null = p_value < alpha
    
    # Bestimme, welches Modell besser ist
    if wins_alg1 > wins_alg2:
        better_alg = model1
    elif wins_alg1 < wins_alg2:
        better_alg = model2
    else:
        better_alg = None  # Kein eindeutiger Gewinner bei einem Unentschieden
    
    # Gib den P-Wert, die Entscheidung über die Nullhypothese und den besseren Algorithmus zurück
    return p_value, reject_null, better_alg
