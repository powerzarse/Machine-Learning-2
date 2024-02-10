import json

# Funktion zum Speichern der Ergebnisse
def save_results_to_json(file_path, results):
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)
