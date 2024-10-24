"""This module implements a model training and evaluation pipeline."""

import mlflow
import numpy as np
from prefect import task
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Function Definitions Section


def rmse_test(model, x_train, y_train, x_test, y_test):
    """Calculate RMSE for a model on test data after training."""
    model.fit(x_train, y_train)
    mlflow.set_tracking_uri("mlruns")
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse * 100


@task(name="train_models_task")
def train_models_task(x_train, y_train, x_test, y_test):
    """Train and evaluate multiple regression models using MLflow tracking."""
    models = [LinearRegression(), Ridge()]
    names = ["LR", "Ridge"]

    best_model = None
    best_model_name = None
    best_rmse = float("inf")

    mlflow.set_experiment("model_evaluation_experiment")

    for model, name in zip(models, names):
        with mlflow.start_run(run_name=name):
            test_rmse = rmse_test(model, x_train, y_train, x_test, y_test)
            print(f"{name}    : RMSE on Test Set = {test_rmse:.6f}")

            # Log model and metrics
            mlflow.log_param("model_name", name)
            mlflow.log_metric("rmse", test_rmse)

            # Track the best model
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_model = model
                best_model_name = name

    if best_model is not None:
        with mlflow.start_run(run_name="best_model"):
            print(f"Best model is {best_model_name} with RMSE {best_rmse:.6f}")
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_rmse", best_rmse)

            # Log the best model
            try:
                mlflow.sklearn.log_model(
                    best_model,
                    artifact_path="best_model",
                    registered_model_name=best_model_name,
                )
            except Exception as e:
                print(f"Error logging the model: {e}")

    return best_model, best_model_name, best_rmse
