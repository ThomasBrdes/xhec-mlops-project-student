"""FastAPI module for ML predictions."""

import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Define your Pydantic model here
class PredictionInput(BaseModel):
    """Data model representing the input features for making predictions."""

    feature_1: float
    feature_2: float
    # Add more features depending on your model's input


class PredictionOutput(BaseModel):
    """Data model representing the prediction output from the ML model."""

    prediction: float


def load_object(file_path: str):
    """Load a pickled object from a specified file path for model usage."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Load the trained model
model = load_object("../modelling/LR.pkl")

# Create FastAPI app instance
app = FastAPI(title="Prediction API", description="API for ML model preds")


@app.get("/")
def home() -> dict:
    """Return health check status to confirm the app is running properly."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOutput, status_code=201)
def predict(payload: PredictionInput) -> PredictionOutput:
    """Predict output using the loaded model based on input features."""
    try:
        # Convert payload to model input format
        features = [[payload.feature_1, payload.feature_2]]

        # Get prediction from model
        prediction = model.predict(features)[0]

        # Return response
        return PredictionOutput(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict err: {str(e)}")
