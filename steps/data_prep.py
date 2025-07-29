from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from zenml import step

@step
def prepare_data(df: pd.DataFrame, use_inference: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame
]:
    df = df.copy()

    input_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[input_features]
    y = df['Outcome']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    input_features_df = pd.DataFrame([input_features], columns=input_features)
    return X_train, X_test, y_train, y_test, input_features_df