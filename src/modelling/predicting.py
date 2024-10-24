def make_predictions(model, x_test):
    """Make predictions using the trained model on the test data."""
    predictions = model.predict(x_test)
    return predictions
