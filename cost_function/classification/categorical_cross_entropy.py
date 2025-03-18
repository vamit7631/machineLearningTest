import numpy as np 
from sklearn.metrics import log_loss


# Example Data for Binary Classification
y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_multi = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])


def categorical_cross_entropy_numpy_mean(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))


###################################################


def categorical_cross_entropy_numpy_sum(y_true, y_pred):
    epsilon = 1e-15
    n = len(y_true)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(np.sum(y_true * np.log(y_pred), axis = 1)) / n


###################################################

def categorical_cross_entropy_sklearn(y_true, y_pred):
    return log_loss(y_true, y_pred)





print("Categorical Cross-Entropy (NumPy mean):", categorical_cross_entropy_numpy_mean(y_true_multi, y_pred_multi))
print("Categorical Cross-Entropy (NumPy sum):", categorical_cross_entropy_numpy_sum(y_true_multi, y_pred_multi))
print("Categorical Cross-Entropy (Using Sklearn):", categorical_cross_entropy_sklearn(y_true_multi, y_pred_multi))