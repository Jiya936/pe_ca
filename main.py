import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Kaggle Titanic Survival Prediction API",
    description="API for predicting passenger survival using trained model parameters",
    version="1.0",
)

# Load saved model artifacts during startup
try:
  model = joblib.load("titanic_model.pkl")
  scaler = joblib.load("scaler.pkl")
except Exception as e:
  raise RuntimeError(f"Error loading model artifacts: {e}")


# Define Pydantic schema for request validation
class TitanicInput(BaseModel):
  Pclass: int = Field(
      ..., example=3, description="Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)"
  )
  Sex: int = Field(
      ..., example=0, description="Sex (0 = male, 1 = female)"
  )
  Age: float = Field(..., example=22.0, description="Passenger Age in years")
  SibSp: int = Field(
      ..., example=1, description="Number of siblings / spouses aboard"
  )
  Parch: int = Field(
      ..., example=0, description="Number of parents / children aboard"
  )
  Fare: float = Field(..., example=7.25, description="Passenger Ticket Fare")


class PredictionOutput(BaseModel):
  survived: int
  prediction_label: str
  survival_probability: float


@app.get("/")
def home():
  return {
      "message": (
          "Welcome to the Titanic API! Go to http://localhost:8000/docs for"
          " Swagger UI."
      )
  }


@app.post("/predict", response_model=PredictionOutput)
def predict_survival(data: TitanicInput):
  try:
    features = np.array([[
        data.Pclass,
        data.Sex,
        data.Age,
        data.SibSp,
        data.Parch,
        data.Fare,
    ]])
    scaled_features = scaler.transform(features)

    pred = int(model.predict(scaled_features)[0])
    prob = float(model.predict_proba(scaled_features)[0][1])
    label = "Survived" if pred == 1 else "Did not survive"

    return {
        "survived": pred,
        "prediction_label": label,
        "survival_probability": round(prob, 4),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))