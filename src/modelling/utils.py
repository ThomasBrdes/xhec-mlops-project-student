"""
This module provides functionality to save a trained model using the pickle
module.
"""

import pickle


def save_model(
    model,
    model_name,
):
    """
    Save a trained model to a file using pickle.

    Args:
        model: The trained model object to be saved.
        model_name (str): The name of the file (without extension) to save the
        model as.

    Returns:
        None
    """
    with open(
        f"{model_name}.pkl",
        "wb",
    ) as f:
        pickle.dump(
            model,
            f,
        )
    print(f"Model saved as {model_name}.pkl")
