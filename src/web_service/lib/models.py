"""
Defines Pydantic models for validating input features and structuring
the outputfor the prediction API.
"""

from pydantic import BaseModel


# Define your input model, match it to what your model expects
class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float
    # Add more features based on your model's input requirements


# Define your output model
class PredictionOutput(BaseModel):
    prediction: float
