"""Pydantic models for validating input and structuring output."""

from pydantic import BaseModel


# Define your input model, match it to what your model expects
class PredictionInput(BaseModel):
    """Model for validating input features for the prediction API."""

    feature_1: float
    feature_2: float
    # Add more features based on your model's input requirements


# Define your output model
class PredictionOutput(BaseModel):
    """Model for structuring the output prediction from the API."""

    prediction: float
