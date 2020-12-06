import matplotlib.pyplot as plt
import numpy as np


class MLP:
    def __init__(self, num_iter):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.num_iter = num_iter

    def sigmoid(self, z):
        return 1 / 1 + np.exp(-z)

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def cost_function(self, p, Y, regularizer=0.8):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + (
                (regularizer / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))))/len(Y)

    def train(self, X, Y, learning_rate=0.02, regularizer=0.8):
        N, D = X.shape
        self.W1 = np.random.randn(D, 40) * 0.01
        self.b1 = np.zeros(40)
        self.W2 = np.random.randn(40, 3) * 0.01
        self.b2 = np.zeros(3)
        self.error_rate = []
        for i in range(self.num_iter):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)
            dZ2 = A2 - Y
            dZ2_dW2 = A1
            dW2 = np.dot(dZ2_dW2.T, dZ2) / len(X) + np.multiply(regularizer, self.W2)/len(X)
            db2 = dZ2 / len(X)
            dW1 = (np.dot(X.T, self.sigmoid_der(Z1) * np.dot(dZ2, self.W2.T))) / len(X) + np.multiply(regularizer,
                                                                                                      self.W1)/len(X)
            db1 = (np.dot(dZ2, self.W2.T) * self.sigmoid_der(Z1)) / len(X)
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1.sum(axis=0)
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2.sum(axis=0)
            j = self.cost_function(A2, Y)
            self.error_rate.append(j)
        self.plot(X, Y)
        print("Accuracy", self.accuracy(A2, Y))

    def accuracy(self, p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return 100 * accuracy / len(p)

    def plot(self, X, Y):
        minX, maxX = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        minY, maxY = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x, y = np.meshgrid(np.arange(minX, maxX, 0.01), np.arange(minY, maxY, 0.01))
        a = np.c_[x.ravel(), y.ravel()]
        Z1 = np.dot(a, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        p = self.softmax(Z2)
        p = np.argmax(p, axis=1)
        p = p.reshape(x.shape)
        #plt.pcolormesh(x, y, p, shading='nearest')
        plt.contourf(x, y, p, cmap=plt.cm.Paired)
        # plt.axis('off')
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='black', cmap=plt.cm.Paired)
        plt.show()


def main():
    with open('spiral_train.dat', 'r') as f:
        data = [i.strip().split(",") for i in f.readlines()]
    data = np.array(data).astype(float)
    X = data[:, : -1]
    Y = data[:, -1]
    X = np.array(X)
    Y = np.eye(3)[np.array(Y).astype(int).reshape(-1)]
    model = MLP(100)
    model.train(X, Y)
    plt.plot(model.error_rate)
    print(model.error_rate[-1])
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
