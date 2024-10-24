"""Module for making predictions using trained machine learning models."""

from prefect import task


@task(name="make_predictions_task")
def make_predictions_task(model, x_test):
    """Make predictions using a trained model on test data.

    Takes a trained model and test features to generate predictions
    for the target variable.

    Args:
        model: A trained machine learning model with predict method.
        x_test (array-like): Test feature set to make predictions on.

    Returns:
        array-like: Predicted values for the test set.
    """
    predictions = model.predict(x_test)
    return predictions
