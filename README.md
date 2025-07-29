**Diabetes Risk Prediction with ZenML and FastAPI**

This project is focused on predicting the risk of diabetes in patients using machine learning techniques. It uses the PIMA Indian Diabetes Dataset and leverages ZenML to build a reproducible ML pipeline, FastAPI to serve the model as an API, and Logfire for logging and monitoring.

**Project Overview**
The pipeline performs the following steps:

Loads and preprocesses the diabetes dataset

Handles class imbalance using SMOTE

Trains a Random Forest Classifier

Evaluates the model based on performance metrics

Generates predictions for new data

Serves the model using a FastAPI endpoint


**Model Deployment**
The trained model is served through a FastAPI application. The /predict/ endpoint allows users to input patient data and receive a prediction on whether the patient is at risk of diabetes, along with the probability score.

**ZenML Artifacts**
ZenML tracks the pipeline artifacts including the trained model (train_model), model evaluation (evaluate_model), and predictions (predict_model). These can be viewed in the ZenML dashboard.

**Logging and Monitoring**
Logfire is integrated to capture logs and track predictions or errors. To activate it, a valid Logfire API key must be added to your .env file.


**Author**
This project was developed by [Adeleye Ayokunle Temitope] to demonstrate end-to-end deployment of a machine learning model using ZenML and FastAPI for a healthcare use case.