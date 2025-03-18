import numpy as np 
import matplotlib.pyplot as plt 

# Step 1: Define Student Data (Study Hours vs Exam Scores)
X = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
y = np.array([52, 55, 60, 65, 68, 72, 75, 78, 82, 85, 88, 91])

# Step 2: Initialize parameters
w = np.random.randn()  # Random weight
b = np.random.randn()  # Random bias
alpha = 0.01 # learning rate
epochs = 1000 # Number of iterations
n = len(X)  # Number of data points

# Step 3: Store loss for visualization
loss_history = []

for i in range(epochs):
    y_pred = w * X + b # Predicted values using formula: ŷ = wX + b
    output = y - y_pred

    # Compute gradients manually
    dw = (-2 / n) * sum(X * output) # Using formula: ∂MSE/∂w = (-2/n) * Σ X(y - ŷ)
    db = (-2 / n) * sum(output) # Using formula: ∂MSE/∂w = (-2/n) * Σ X(y - ŷ)


    # Update parameters using gradient descent rule
    w -= alpha * dw # Using formula: w = w - α * ∂MSE/∂w 
    b -= alpha * db # Using formula: b = b - α * ∂MSE/∂b

    mse = np.mean(output ** 2) # Using formula: MSE = (1/n) * Σ (y - ŷ)²

    loss_history.append(mse)

    # Print loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {mse}")

# Step 5: Final parameters
print(f"\n Final Weight : {w : .4f}, final Bias : {b: .4f}")

# Step 6: Plot Loss Curve
plt.plot(range(epochs), loss_history, label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Loss Reduction Over time")
plt.legend()
plt.show()

# Step 7: Plot Regression Line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, w * X + b, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()