# steps/data_loader.py

from zenml import step
import pandas as pd
from typing import Optional, Dict

@step
def load_data(
    input_data: Optional[Dict] = None,
    use_inference: bool = False
) -> pd.DataFrame:
    """
    Load the dataset either from CSV (for training) or from input dictionary (for live inference).

    Args:
        input_data (Optional[Dict]): Dictionary with input features for prediction.
        use_inference (bool): If True, use input_data; else load from CSV.

    Returns:
        pd.DataFrame: DataFrame ready for processing.
    """
    if use_inference:
        if input_data is None:
            raise ValueError("Input data must be provided for inference.")
        df = pd.DataFrame([input_data])
        return df
    else:
        df = pd.read_csv("diabetes.csv")
        return df

# Optional: for local test run (not used in pipeline directly)
if __name__ == "__main__":
    input_data = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 79,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 32
    }
    df = load_data(input_data=input_data, use_inference=True)
    print(df)