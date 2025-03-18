import numpy as np 
from sklearn.metrics import log_loss


# Example Data for Binary Classification
y_true_binary = np.array([1, 0, 1, 1, 0])
y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Probability scores



def binary_cross_entropy_numpy_mean(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


###################################################


def binary_cross_entropy_numpy_sum(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    n = len(y_true)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n

###################################################


# Scikit-learn method
def binary_cross_entropy_sklearn(y_true, y_pred):
    return log_loss(y_true, y_pred)


print("Binary Cross-Entropy (NumPy mean):", binary_cross_entropy_numpy_mean(y_true_binary, y_pred_binary))
print("Binary Cross-Entropy (NumPy sum):", binary_cross_entropy_numpy_sum(y_true_binary, y_pred_binary))
print("Binary Cross-Entropy (sklearn):", binary_cross_entropy_sklearn(y_true_binary, y_pred_binary))