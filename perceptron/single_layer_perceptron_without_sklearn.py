import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1) # + 1 for bias 
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self,x):
        return self.activation(np.dot(x, self.weights[1:]) + self.weights[0])

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y): 
                pred = self.predict(xi)
                update = self.lr * (target - pred)
                self.weights[1:] += update * xi
                self.weights[0] += update



X = np.array([[0,0], [0,1], [1,0],[1,1]]) # AND GATE
y = np.array([0,0,0,1]) # AND GATE OUTPUT

perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

print([perceptron.predict(x) for x in X]) # Output: [0, 0, 0, 1]