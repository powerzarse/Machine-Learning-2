import os
import numpy as np
import json

def save_history_to_json(history, filename, path):
    """
    Save the training history to a JSON file.

    Args:
    history: The training history object (either a dictionary or a Keras History object).
    filename (str): The name of the JSON file.
    path (str): The path to save the JSON file.

    Returns:
    bool: True if saving was successful, False otherwise.
    """
    full_path = os.path.join(path, filename)
    try:
        with open(full_path, 'w') as file:
            if isinstance(history, dict):
                # If 'history' is already a dictionary
                history_dict = {key: np.array(val).tolist() for key, val in history.items()}
            else:
                # If 'history' is a Keras History object
                history_dict = {key: np.array(val).tolist() for key, val in history.history.items()}
            json.dump(history_dict, file, indent=4)
        print(f"History successfully saved to: {full_path}")
        return True
    except Exception as e:
        print(f"Error occurred while saving history: {e}")
        return False
