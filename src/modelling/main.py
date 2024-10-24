"""Main module for training the predictive model."""

import pandas as pd
from predicting import make_predictions_task
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from preprocessing import preprocess_data_task
from training import train_models_task
from utils import save_model_task


@task(name="load_data", retries=3, retry_delay_seconds=60)
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


@flow(name="train_model_flow")
def main():
    """Train a model using the data and save the model (pickle).

    This function loads the dataset, preprocesses it, trains multiple models,
    selects the best performing one, and saves it to disk. Optionally makes
    predictions using the trained model.
    """
    # Load your data
    df = load_data("../../data/abalone.csv")

    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data_task(df)

    # Train models and get the best model
    best_model, best_model_name, best_rmse = train_models_task(
        x_train, y_train, x_test, y_test
    )

    save_model_task(best_model, best_model_name)

    # Make predictions (optional)
    predictions = make_predictions_task(best_model, x_test)
    print(predictions)  # Use predictions to avoid F841


def create_deployment():
    """Create a deployment for regular model retraining."""
    deployment = Deployment.build_from_flow(
        flow=main,
        name="train_model_deployment",
        schedule=(CronSchedule(cron="0 0 * * 0")),
        version="1",
        tags=["ml", "training"],
    )
    deployment.apply()


if __name__ == "__main__":
    # Create deployment
    create_deployment()
    # Run the flow
    main()
