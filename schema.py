# schema.py

from pydantic import BaseModel, Field
from typing import Optional


class PatientInput(BaseModel):
    Pregnancies: float = Field(..., example=2)
    Glucose: float = Field(..., example=120)
    BloodPressure: float = Field(..., example=70)
    SkinThickness: float = Field(..., example=20)
    Insulin: float = Field(..., example=80)
    BMI: float = Field(..., example=25.0)
    DiabetesPedigreeFunction: float = Field(..., example=0.5)
    Age: float = Field(..., example=33)


class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_proba: float