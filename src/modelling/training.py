"""Module for training and evaluating models with MLflow tracking."""

import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def rmse_test(model, x_train, y_train, x_test, y_test):
    """Calculate RMSE for a model on test data after training.

    Train the given model on the training data and evaluate it using RMSE on
    the test data.

    Args:
        model: A regression model that supports the fit and predict methods.
        x_train (array-like): Training feature set.
        y_train (array-like): Training target values.
        x_test (array-like): Test feature set.
        y_test (array-like): True target values for the test set.

    Returns:
        float: The RMSE of the model predictions on the test set scaled by 100.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse * 100


def train_models(x_train, y_train, x_test, y_test):
    """Train and evaluate multiple regression models using MLflow tracking.

    This function evaluates several regression models, logs their RMSE metrics,
    and tracks the best performing model based on the RMSE.

    Args:
        x_train (array-like): Training feature set.
        y_train (array-like): Training target values.
        x_test (array-like): Test feature set.
        y_test (array-like): True target values for the test set.

    Returns:
        tuple: A tuple containing the best model, the name of the best model,
            and the best RMSE value.
    """
    models = [LinearRegression(), Ridge()]
    names = ["LR", "Ridge", "SVR", "RF", "GB", "KNN"]
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
            mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

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
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="best_model",
                registered_model_name=best_model_name,
            )

    return best_model, best_model_name, best_rmse
