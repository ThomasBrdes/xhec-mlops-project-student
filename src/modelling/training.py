"""
This module contains functions to train and evaluate regression models using
RMSE and log their performance with MLflow.
"""

import mlflow
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def rmse_test(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train the given model on the training data and evaluate it using RMSE on
    the test data.

    Args:
        model: A regression model that supports the fit and predict methods.
        X_train (array-like): Training feature set.
        y_train (array-like): Training target values.
        X_test (array-like): Test feature set.
        y_test (array-like): True target values for the test set.

    Returns:
        float: The RMSE of the model predictions on the
        test set, scaled by 100.
    """
    model.fit(
        X_train,
        y_train,
    )
    y_pred = model.predict(X_test)
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )
    return rmse * 100


def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train multiple regression models and log their performance using MLflow.

    This function evaluates several regression models, logs their RMSE metrics
    , and tracks
    the best performing model based on the RMSE.

    Args:
        X_train (array-like): Training feature set.
        y_train (array-like): Training target values.
        X_test (array-like): Test feature set.
        y_test (array-like): True target values for the test set.

    Returns:
        tuple: A tuple containing the best model, the name of the best model,
               and the best RMSE value.
    """
    models = [
        LinearRegression(),
        Ridge(),
        SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        KNeighborsRegressor(n_neighbors=4),
    ]

    names = [
        "LR",
        "Ridge",
        "SVR",
        "RF",
        "GB",
        "KNN",
    ]
    best_model = None
    best_model_name = None
    best_rmse = float("inf")

    mlflow.set_experiment("model_evaluation_experiment")

    for (
        model,
        name,
    ) in zip(
        models,
        names,
    ):
        with mlflow.start_run(run_name=name):
            test_rmse = rmse_test(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            print(f"{name}    : RMSE on Test Set = {test_rmse:.6f}")

            # Log model and metrics
            mlflow.log_param(
                "model_name",
                name,
            )
            mlflow.log_metric(
                "rmse",
                test_rmse,
            )
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"{name}_model",
            )

            # Track the best model
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_model = model
                best_model_name = name

    if best_model is not None:
        with mlflow.start_run(run_name="best_model"):
            print(f"Best model is {best_model_name} with RMSE {best_rmse:.6f}")
            mlflow.log_param(
                "best_model",
                best_model_name,
            )
            mlflow.log_metric(
                "best_rmse",
                best_rmse,
            )
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="best_model",
                registered_model_name=best_model_name,
            )

    return (
        best_model,
        best_model_name,
        best_rmse,
    )
