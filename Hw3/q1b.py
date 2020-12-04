import numpy as np
import matplotlib.pyplot as plt


class MaximumEntropyClassifier:
    def __init__(self, num_iter):
        self.W = None
        self.b = None
        self.num_iter = num_iter

    def feed_forward(self, X, w, b):
        f = np.dot(X, w) + b
        f = np.exp(f)
        denominator = f.sum(axis=1)
        for i in range(len(denominator)):
            f[i, :] /= denominator[i]
        return f

    def cost_function(self, p, Y, w, regularizer=0.7):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + ((regularizer / 2) * np.sum(np.square(w)))

    def gradient(self, X, Y, p, regularizer=0.7):
        dw = (np.dot(X.T, (p - Y)) / len(X)) + regularizer * self.W
        db = sum(p - Y) / len(X)
        return dw, db

    def train(self, X, Y, learning_rate=0.01):
        self.error_rate = []
        N, D = X.shape
        k = len(np.unique(Y))
        self.W = np.random.randn(D, k)
        self.b = np.random.randn(k)
        Y_hotencode = np.eye(3)[np.array(Y).astype(int).reshape(-1)]
        for i in range(self.num_iter):
            p = self.feed_forward(X, self.W, self.b)
            j = self.cost_function(p, Y_hotencode, self.W)
            dw, db = self.gradient(X, Y_hotencode, p)
            self.W -= learning_rate * dw
            self.b -= learning_rate * db
            self.error_rate.append(j)
        print(self.accuracy(p, Y_hotencode))
        self.plot(X, Y)

    def accuracy(self, p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return 100 * (accuracy / len(p))

    def plot(self, X, Y):
        minX, maxX = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        minY, maxY = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x, y = np.meshgrid(np.arange(minX, maxX, 0.01), np.arange(minY, maxY, 0.01))
        a = np.c_[x.ravel(), y.ravel()]
        p = self.feed_forward(a, self.W, self.b)
        p = np.argmax(p, axis=1)
        p = p.reshape(x.shape)
        plt.pcolormesh(x, y, p)
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='black')
        plt.show()

def main():
    with open('spiral_train.dat', 'r') as f:
        data = [i.strip().split(",") for i in f.readlines()]
    data = np.array(data).astype(float)
    np.random.shuffle(data)
    X = data[:, : -1]
    Y = data[:, -1]
    X = np.array(X)
    # print(X.shape)
    # print(Y)
    model = MaximumEntropyClassifier(1000)
    model.train(X, Y)
    plt.plot(model.error_rate)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
