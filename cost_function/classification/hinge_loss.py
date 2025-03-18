import numpy as np
from sklearn.metrics import hinge_loss

y_true_binary = np.array([1, 0, 1, 1, 0])
y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Probability scores

# NumPy mean method
def hinge_loss_numpy_mean(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


###################################################

# NumPy sum method
def hinge_loss_numpy_sum(y_true, y_pred):
    n = len(y_true)
    return np.sum(np.maximum(0, 1 - y_true * y_pred)) / n


###################################################

# Scikit-learn method
def hinge_loss_sklearn(y_true, y_pred):
    return hinge_loss(y_true, y_pred)



print("Hinge Loss (NumPy mean):", hinge_loss_numpy_mean(y_true_binary, y_pred_binary))
print("Hinge Loss (NumPy sum):", hinge_loss_numpy_sum(y_true_binary, y_pred_binary))
print("Hinge Loss (Using Sklearn):", hinge_loss_sklearn(y_true_binary, y_pred_binary))