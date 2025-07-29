from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from zenml.client import Client
import logfire

# Initialize Logfire
logfire.configure()

# Define FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# Load ZenML pipeline output model
client = Client()
model = client.get_pipeline("diabetes_pipeline").get_step("train_model").outputs["output"]

# Define request body
class PatientInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(input: PatientInput):
    # Convert input to array
    input_array = np.array([[input.Pregnancies, input.Glucose, input.BloodPressure, input.SkinThickness,
                             input.Insulin, input.BMI, input.DiabetesPedigreeFunction, input.Age]])
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)[0][1]
    
    logfire.log("prediction", {"class": int(prediction[0]), "probability": float(proba)})

    return {
        "predicted_class": int(prediction[0]),
        "probability_of_diabetes": float(proba)
    }