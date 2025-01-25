
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def train_model(data: pd.DataFrame, target_column: str):
    """
    Train a logistic regression model and save it.

    Args:
        data (pd.DataFrame): The training dataset.
        target_column (str): The name of the target column.

    Returns:
        model: Trained logistic regression model.
        auc: AUC-ROC score.
        fpr: False positive rates from ROC curve.
        tpr: True positive rates from ROC curve.
        thresholds: Thresholds from ROC curve.
        y_true: Ground truth labels.
        y_pred_prob: Predicted probabilities for the target column.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    model = LogisticRegression()
    model.fit(X, y)

    # Save the trained model to a file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Calculate AUC-ROC metrics
    y_pred_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

    return model, auc, fpr, tpr, thresholds, y, y_pred_prob


def predict(data: pd.DataFrame, model_path="model.pkl"):
    """
    Predict using a saved logistic regression model.

    Args:
        data (pd.DataFrame): The test dataset.
        model_path (str): Path to the saved model file.

    Returns:
        predictions: Predicted class labels.
        probabilities: Predicted probabilities for the positive class.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict probabilities and classes
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate precision, recall, accuracy, and F1-score based on a given threshold.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold for converting probabilities to class labels.

    Returns:
        precision: Precision score.
        recall: Recall score.
        accuracy: Accuracy score.
        f1: F1-score.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, accuracy, f1
