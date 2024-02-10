from tensorflow import keras

def check_compile(model):
    """
    Check if the model is compiled.

    Args:
    model: The Keras model to check.

    Raises:
    ValueError: If the model is not compiled.
    """
    if not model.optimizer:
        raise ValueError("Original model must be compiled before using this function.")
    else:
        print('Alright. Model is already compiled and trained')
