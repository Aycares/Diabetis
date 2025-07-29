from zenml import step
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import ClassifierMixin
import pandas as pd
from typing import Optional

@step
def evaluate_model(
    model: Optional[ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_inference: bool = False
) -> Optional[str]:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        use_inference: Flag to skip evaluation during inference.

    Returns:
        Evaluation summary as a string, or None during inference.
    """
    if use_inference or model is None:
        return None

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    summary = f"\nAccuracy: {acc:.4f}\n\n{report}"
    print(summary)
    return summary