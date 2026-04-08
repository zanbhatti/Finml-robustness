import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error:
    average absolute difference between true values and predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error:
    square root of average squared difference
    penalizes larger errors more heavily than MAE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))