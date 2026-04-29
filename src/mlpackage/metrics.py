import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy for classification tasks.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth (correct) labels.
    y_pred : numpy.ndarray
        Predicted labels.
    """
    return np.mean(y_true == y_pred)

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error for regression tasks.
    """
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    """
    Calculate the R^2 (coefficient of determination) for regression tasks.
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)