def make_predictions(model, X_test):
    """Make predictions using the trained model on the test data."""
    predictions = model.predict(X_test)
    return predictions
