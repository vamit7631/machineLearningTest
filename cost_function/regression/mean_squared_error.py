import numpy as np

y_actual = np.array([3, -0.5, 2, 7])
y_predicted = np.array([2.5, 0.0, 2, 8])

# Using NumPy's mean() method
def mse_using_mean(y_true, y_pred):
    """
    MSE Formula using mean():
    
    MSE = mean((y_true - y_pred)²)
    
    """
    return np.mean((y_true - y_pred) ** 2)


###################################################


# Using NumPy's sum() method
def mse_using_sum(y_true, y_pred):
    """
    MSE Formula using sum():
    
    MSE = (1/n) * sum((y_true - y_pred)²)
    
    """
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n


###################################################


# Without NumPy (Using Only Python)


def mse_without_numpy(y_true, y_pred):
    """
    MSE Formula without NumPy:
    
    MSE = (1/n) * Σ (y_true[i] - y_pred[i])²
    
    """
    n = len(y_true)
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors)/n
    return mse



###################################################


# Using sklearn library method

from sklearn.metrics import mean_squared_error

mse_sklearn = mean_squared_error(y_actual, y_predicted)
print(f"MSE using Scikit-Learn: {mse_sklearn:.4f}")





mse1 = mse_using_mean(y_actual, y_predicted)
mse2 = mse_using_sum(y_actual, y_predicted)
mse3 = mse_without_numpy(y_actual.tolist(), y_predicted.tolist()) 

print(f"MSE using mean(): {mse1:.4f}")
print(f"MSE using sum(): {mse2:.4f}")
print(f"MSE without NumPy: {mse3:.4f}")