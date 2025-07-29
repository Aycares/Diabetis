from zenml import step
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
import pandas as pd

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_inference: bool = False
) -> Optional[ClassifierMixin]:
    """
    Train a Random Forest Classifier on the given training data.
    If `use_inference` is True, skip training and return None.

    Args:
        X_train: Training features.
        y_train: Training labels.
        use_inference: Flag to skip training.

    Returns:
        Trained model or None.
    """
    if use_inference:
        return None

    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    return clf  # return the model object directly