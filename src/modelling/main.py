"""Main module for training the predictive model."""

import pandas as pd
from predicting import make_predictions
from preprocessing import preprocess_data
from training import train_models
from utils import save_model


def main():
    """Train a model using the data and save the model (pickle).

    This function loads the dataset, preprocesses it, trains multiple models,
    selects the best performing one, and saves it to disk. Optionally makes
    predictions using the trained model.
    """
    # Load your data
    # Update with your actual data file path
    df = pd.read_csv("../../data/abalone.csv")

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train models and get the best model
    best_model, best_model_name, best_rmse = train_models(
        X_train, y_train, X_test, y_test
    )

    save_model(
        best_model,
        best_model_name,
    )

    # Make predictions (optional)
    predictions = make_predictions(best_model, X_test)
    print(predictions)  # Use predictions to avoid F841


if __name__ == "__main__":
    main()
