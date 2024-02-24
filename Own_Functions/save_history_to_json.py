import os
import json
import numpy as np

def save_history_to_json(history, filename, path):
    """
    Handles the training history: saves it if provided; otherwise, checks if a history file already exists.

    Args:
    history: The training history object (either a dictionary, a Keras History object, or None).
    filename (str): The name of the JSON file.
    path (str): The path to save the JSON file or check for its existence.

    Returns:
    None
    """
    full_path = os.path.join(path, filename)
    
    # Wenn eine Historie vorhanden ist, speichere sie
    if history is not None:
        try:
            # Konvertiere die Historie in ein speicherbares Format
            history_dict = history if isinstance(history, dict) else history.history
            history_dict = {k: np.array(v).tolist() for k, v in history_dict.items()}
            
            # Speichere die Historie in einer JSON-Datei
            with open(full_path, 'w') as file:
                json.dump(history_dict, file, indent=4)
            print(f"Training history saved to: {full_path}")
        except Exception as e:
            print(f"Error occurred while saving history: {e}")
    
    # Wenn keine Historie vorhanden ist, überprüfe, ob eine Datei existiert
    elif os.path.exists(full_path):
        print(f"Training history file already exists at: {full_path}.")
    
    # Wenn keine Historie vorhanden ist und keine Datei existiert, informiere den Benutzer
    else:
        print("No training history to save, and no existing history file found.")
