import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, y_pred):
    m = len(y)
    return - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression_gd(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    losses = []

    for epoch in range(epochs):
        linear_model = np.dot(X, theta) + bias
        y_pred = sigmoid(linear_model)

        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        theta -= lr * dw
        bias -= lr * db

        loss = binary_cross_entropy(y, y_pred)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return theta, bias, losses

# Sample dataset
X = np.array([[2, 3], [3, 5], [5, 8], [7, 10], [1, 2], [8, 12], [9, 15]])
y = np.array([0, 0, 0, 1, 0, 1, 1])

theta, bias, losses = logistic_regression_gd(X, y)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()


def predict(X, theta, bias):
    linear_model = np.dot(X, theta) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]

# Test predictions
X_test = np.array([[4, 6], [6, 9], [10, 14]])
predictions = predict(X_test, theta, bias)
print("Predictions:", predictions)
