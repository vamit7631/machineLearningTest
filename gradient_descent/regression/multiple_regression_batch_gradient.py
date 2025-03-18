import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Student Data (Multiple Features)
X = np.array([
    [1.5, 8, 50], [2.0, 7, 55], [2.5, 6, 60], [3.0, 6, 65], [3.5, 5, 68], [4.0, 5, 72], 
    [4.5, 5, 75], [5.0, 4, 78], [5.5, 4, 82], [6.0, 4, 85], [6.5, 3, 88], [7.0, 3, 91]
])

y = np.array([52, 55, 60, 65, 68, 72, 75, 78, 82, 85, 88, 91])  # Final Exam Scores

# Normalize X (Feature Scaling)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std  # Normalize y for better convergence

# Step 2: Initialize parameters
m, n = X.shape  # m: number of samples, n: number of features
w = np.random.randn(n) * 0.01  # Small random initialization
b = np.random.randn() * 0.01
alpha = 0.01  # Increased learning rate for faster convergence
epochs = 1000  # Number of iterations

# Step 3: Store loss for visualization
loss_history = []

for i in range(epochs):
    y_pred = np.sum(X_norm * w, axis=1) + b  # Linear model
    output = y_norm - y_pred  # Compute error

    # Compute gradients manually
    dw = (-2 / m) * np.sum(output.reshape(-1, 1) * X_norm, axis=0)
    db = (-2 / m) * np.sum(output)

    # Update parameters using gradient descent rule
    w -= alpha * dw
    b -= alpha * db

    mse = np.mean(output ** 2)  # Compute Mean Squared Error (MSE)
    loss_history.append(mse)

    # Print loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i} : loss = {mse:.4f}")

# Step 5: Final parameters
print(f"\nFinal Weights: {w}, Final Bias: {b:.4f}")

# Step 6: Plot Loss Curve
plt.plot(range(epochs), loss_history, label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Loss Reduction Over Time")
plt.legend()
plt.show()

# Prediction for new student
new_student = np.array([[6, 5, 80]])
new_student_norm = (new_student - X_mean) / X_std  # Normalize input

predicted_score_norm = np.sum(new_student_norm * w) + b  # Prediction in normalized scale
predicted_score = (predicted_score_norm * y_std) + y_mean  # Denormalize

print(f"Predicted Final Examination Score: {predicted_score:.2f}")
