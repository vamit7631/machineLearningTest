import numpy as np

class XORMLP:
    def __init__(self, input_size = 2, hidden_size = 2, output_size = 1, lr = 0.1, seed = 1):
        np.random.seed(seed)
        self.lr = lr

        # Weight initialization
        self.W1 = np.random.uniform(size=(input_size, hidden_size))
        self.b1 = np.random.uniform(size=(1, hidden_size))
        self.W2 = np.random.uniform(size=(hidden_size, output_size))
        self.b2 = np.random.uniform(size=(1, output_size))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)
    
        return self.output

    def backward(self, X, y):
        error = y - self.output
        d_output = error * self.sigmoid_derivative(self.output)

        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += self.a1.T.dot(d_output) * self.lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return np.round(self.forward(X))


# Example usage
if __name__ == "__main__":
    # XOR inputs and outputs
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train model
    model = XORMLP()
    model.train(X, y)

    # Predictions
    print("Predictions:")
    print(model.predict(X))

# output [[0.] [1.] [1.] [0.]]