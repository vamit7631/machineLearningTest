import numpy as np

y_actual = np.array([3, -0.5, 2, 7])
y_predicted = np.array([2.5, 0.0, 2, 8])

# Using NumPy's mean() method

def mae_using_mean(y_true, y_pred):
    """    
    Formula:
    MAE = mean(|y_true - y_pred|)

    """
    return np.mean(np.abs(y_true - y_pred))


###################################################

# Using NumPy's sum() method
def mae_using_sum(y_true, y_pred):
    """    
    Formula:
    MAE = (1/n) * sum(|y_true - y_pred|)

    """
    n = len(y_true)
    return np.sum(np.abs(y_true - y_pred)) / n


###################################################


# Without NumPy (Using Only Python)
def mae_without_numpy(y_true, y_pred):
    """
    MAE Formula without NumPy:
    
    MAE = (1/n) * sum(|y_true - y_pred|)
    
    """
    n = len(y_true)
    absolute_errors = [abs(y_true[i] - y_pred[i]) for i in range(n)]
    return sum(absolute_errors) / n 


###################################################


# Using sklearn library method

from sklearn.metrics import mean_absolute_error

mae_sklearn = mean_absolute_error(y_actual, y_predicted)
print(f"MAE using Scikit-Learn: {mae_sklearn:.4f}")


mae1 = mae_using_mean(y_actual, y_predicted)
mae2 = mae_using_sum(y_actual, y_predicted)
mae3 = mae_without_numpy(y_actual.tolist(), y_predicted.tolist())
print(f"MAE using mean(): {mae1:.4f}")
print(f"MAE using sum(): {mae2:.4f}")
print(f"MAE without numpy: {mae3:.4f}")
