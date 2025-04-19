import numpy as np 
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, lr = 0.1, epochs = 10):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0; 

    def predict(self, x):
        return self.activation(np.dot(x, self.weights[1:]) + self.weights[0])

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X,y):
                pred = self.predict(xi)
                update = self.lr * (target - pred)
                self.weights[1:] += update * xi
                self.weights[0] += update                


X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1, 1, 0])

perceptron = Perceptron(input_size = 2)
perceptron.train(X, Y)

print([perceptron.predict(x) for x in X]) # Output: [0, 0, 0, 1]



def plot_decision_boundary(model, X, y):
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), 
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.predict(point) for point in grid])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=100)
    plt.title("Decision Boundary - Perceptron (XOR Gate)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.show()

plot_decision_boundary(perceptron, X, Y)


