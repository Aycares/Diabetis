from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import preprocess_data
from steps.data_training import train_model
from steps.evaluate_model import evaluate_model
from steps.predict_model import predict_model

@pipeline
def diabetes_pipeline(use_inference: bool = False, input_data: dict = None):
    df = load_data(input_data=input_data, use_inference=use_inference)
    X_train, X_test, y_train, y_test, input_features = preprocess_data(
        df=df, input_data=input_data, use_inference=use_inference
    )
    model = train_model(X_train=X_train, y_train=y_train, use_inference=use_inference)
    evaluate_model(model=model, X_test=X_test, y_test=y_test, use_inference=use_inference)
    predict_model(model=model, input_data=input_features, use_inference=use_inference)