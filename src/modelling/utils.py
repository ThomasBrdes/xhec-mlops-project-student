"""Module for saving trained models using pickle serialization."""

import pickle

from prefect import task


@task(name="save_model_task")
def save_model_task(model, model_name):
    """Save a trained model to a file using pickle.

    Args:
        model: The trained model object to be saved.
        model_name (str): The name of the file (without extension) to save the
            model as.

    Returns:
        None
    """
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_name}.pkl")
