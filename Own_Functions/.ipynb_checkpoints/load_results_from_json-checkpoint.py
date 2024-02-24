import os
import json

default_path = '/Users/linuszarse/Documents/UNI/Master-Uni Potsdam/3. Semester/Machine Learning 2/Klausurprojekt/Cross_Validation_Results/'

def load_results_from_json(file_name, path=default_path):
    """Lädt die Ergebnisse aus einer JSON-Datei."""
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        print(f"Die Datei '{file_name}' existiert nicht. Keine Daten zum Laden.")
        return None
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data