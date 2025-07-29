from zenml import step
from sklearn.base import ClassifierMixin
import pandas as pd
from typing import Optional, Tuple

@step
def predict_model(
    model: Optional[ClassifierMixin],
    input_data: Optional[pd.DataFrame] = None,
    use_inference: bool = False
) -> Optional[Tuple[int, float]]:
    """
    Predict the class and probability for a new patient using the trained model.

    Args:
        model: Trained classifier.
        input_data: Single-row input as a pandas DataFrame.
        use_inference: If True, perform prediction. Otherwise, skip.

    Returns:
        Tuple of (predicted class, predicted probability), or None if not inference.
    """
    if not use_inference or model is None or input_data is None:
        return None

    predicted_class = int(model.predict(input_data)[0])
    predicted_proba = float(model.predict_proba(input_data)[0][1])  # Probability of class 1 (at risk)

    print(f"\nPredicted Class: {predicted_class} | Probability: {predicted_proba:.4f}")
    return predicted_class, predicted_proba