from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import prepare_data
from steps.data_training import train_model
from steps.evaluate_model import evaluate_model
from steps.predict_model import predict_model
import numpy as np

# Define pipeline
@pipeline
def diabetes_pipeline(use_inference: bool = False, input_data: dict = None):
    df = load_data(input_data=input_data, use_inference=use_inference)
    X_train, X_test, y_train, y_test, input_features = prepare_data(
        df=df, use_inference=use_inference
    )
    model = train_model(X_train=X_train, y_train=y_train, use_inference=use_inference)
    evaluate_model(model=model, X_test=X_test, y_test=y_test, use_inference=use_inference)
    predict_model(model=model, input_data=input_features, use_inference=use_inference)

# Local training and evaluation
if __name__ == "__main__":
    # For full training and evaluation
    diabetes_pipeline()

    # For live endpoint inference
    # diabetes_pipeline(
    #     use_inference=True,
    #     input_data={
    #         "Pregnancies": 2,
    #         "Glucose": 120,
    #         "BloodPressure": 70,
    #         "SkinThickness": 20,
    #         "Insulin": 79,
    #         "BMI": 25.0,
    #         "DiabetesPedigreeFunction": 0.5,
    #         "Age": 32
    #     }
    # )