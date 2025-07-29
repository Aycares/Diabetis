import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from schema import PatientInput


def apply_smote(X: pd.DataFrame, y: pd.Series):
    """
    Apply SMOTE to balance the dataset.
    
    Args:
        X: Features DataFrame.
        y: Target labels Series.
    
    Returns:
        Tuple of balanced (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def convert_input_to_numpy(input_data: PatientInput) -> np.ndarray:
    """
    Convert PatientInput schema to a NumPy array suitable for prediction.
    
    Args:
        input_data: PatientInput object.
    
    Returns:
        NumPy array of shape (1, n_features).
    """
    return np.array([[input_data.Pregnancies,
                      input_data.Glucose,
                      input_data.BloodPressure,
                      input_data.SkinThickness,
                      input_data.Insulin,
                      input_data.BMI,
                      input_data.DiabetesPedigreeFunction,
                      input_data.Age]])