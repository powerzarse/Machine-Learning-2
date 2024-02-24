from scipy.stats import binom

def sign_test(results_alg1, results_alg2, alpha=0.05):
    # Paare von Ergebnissen für beide Algorithmen erstellen
    paired_results = list(zip(results_alg1, results_alg2))
    
    # Zähle wie oft Algorithmus 1 besser ist als Algorithmus 2
    wins_alg1 = sum(1 for a1, a2 in paired_results if a1 > a2)
    
    # Zähle wie oft Algorithmus 2 besser ist als Algorithmus 1
    wins_alg2 = sum(1 for a1, a2 in paired_results if a1 < a2)
    
    # Zähle wie oft beide Algorithmen die gleiche Leistung haben
    ties = sum(1 for a1, a2 in paired_results if a1 == a2)
    
    # Berechne die Gesamtzahl der Fälle ohne Unentschieden für den Sign-Test
    n = len(paired_results) - ties

    # Wenn alle Vergleiche Unentschieden sind, ist der P-Wert 1
    if n == 0:
        return 1.0, False, "kein eindeutiger Gewinner"
    
    # Bestimme die maximale Anzahl von Gewinnen zwischen den beiden Algorithmen
    T = max(wins_alg1, wins_alg2)
    
    # Berechne den P-Wert basierend auf der Binomialverteilung
    # Multipliziere mit 2 für den zweiseitigen Test (Test in beide Richtungen)
    p_value = 2 * (1 - binom.cdf(T - 1, n, 0.5))
    
    # Entscheide, ob die Nullhypothese aufgrund des berechneten P-Wertes abgelehnt wird
    reject_null = p_value < alpha
    
    # Bestimme, welcher Algorithmus besser ist, basierend auf der Anzahl der Gewinne
    # Wenn beide Algorithmen gleich oft gewinnen, gibt es keinen eindeutigen Gewinner
    better_alg = "Algorithmus 1" if wins_alg1 > len(paired_results) - wins_alg1 - ties else "Algorithmus 2"
    if wins_alg1 == len(paired_results) - wins_alg1 - ties:
        better_alg = "kein eindeutiger Gewinner"
    
    # Gib den P-Wert, die Entscheidung über die Nullhypothese und den besseren Algorithmus zurück
    return p_value, reject_null, better_alg