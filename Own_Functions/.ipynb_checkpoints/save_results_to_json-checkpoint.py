import os
import json

default_path = '/Users/linuszarse/Documents/UNI/Master-Uni Potsdam/3. Semester/Machine Learning 2/Klausurprojekt/Cross_Validation_Results/'

def save_results_to_json(file_name, results, path=default_path):
    """
    Save results to a JSON file.

    Args:
    file_name (str): The name of the JSON file.
    results: The results to be saved.
    path (str): The path to save the JSON file.
    """
    full_path = os.path.join(path, file_name)
    
    if os.path.exists(full_path):
        print(f"Die Datei '{file_name}' existiert bereits im Verzeichnis '{path}'. Speichern wird übersprungen.")
    else:
        with open(full_path, 'w') as file:
            json.dump(results, file, indent=4)
        print(f"Ergebnisse wurden in '{file_name}' erfolgreich gespeichert.")
